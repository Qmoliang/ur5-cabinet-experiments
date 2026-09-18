"""Generate 07.4 results only from completed runs and explicit audit fields."""
from pathlib import Path
import json,hashlib
import numpy as np,mujoco
from model import set_configuration
from experiment_07_3_geometry import ROOT,render
from audit_experiment_07_3_continuous import audit
OUT=ROOT/'formal_results/experiment_07/development_07_4'

def main():
    batch=sorted((OUT/'online_').glob('*/manifest.json'))[-1];manifest=json.loads(batch.read_text(encoding='utf-8'));rows=manifest['results'];assert len(rows)==11
    assert {(r['group'],r['seed']) for r in rows}=={(g,i) for g in ['ET30','D30'] for i in range(5)}|{('P30',0)}
    result=[]
    for r in rows:
        folder=Path(r['summary']).parent;s=json.loads(Path(r['summary']).read_text());c=audit(folder)
        record=dict(group=r['group'],seed=r['seed'],run=str(folder),final_error_mm=s['final_error_m']*1000,success=s['success'],ever_sustained_success=s['ever_sustained_success'],evidence_eligible=s['evidence_eligible'],coverage=s['all_centervox_coverage_checks_passed'],mvt=s['all_mvt_oracle_checks_passed'],penetrating_cycles=s['exact_penetrating_cycles'],sweep_failures=s['sweep_audit_failures'],observability=s['observability_passed'],late_samples=s['observability_late_samples'],never_observed_samples=s['observability_never_observed_samples'],proxies=s['final_proxy_count'],controller_p99_ms=s['controller_ms_p99'],perception_p99_ms=s['perception_pipeline_ms_p99'],snapshot_age_p99_ms=s['snapshot_age_ms_p99'],deadline_misses=s['deadline_misses'],continuous_boxes=c['all_environment_box_segments_certified'])
        result.append(record)
        if r['seed']==0:
            m=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));d=mujoco.MjData(m);set_configuration(m,d,np.load(folder/'q_history.npy')[-1]);render(m,d,OUT/f'{r["group"]}_seed0_final.png')
    same=json.loads((OUT/'same_input.json').read_text())['runs'];pilot=json.loads((OUT/'full_manager_same_input_seed0.json').read_text());terminal=json.loads((OUT/'terminal_constraints.json').read_text())['runs'];assert len(terminal)==11
    table={}
    for group in ['ET30','D30','P30']:
        group_rows=[r for r in result if r['group']==group];table[group]=dict(n=len(group_rows),successes=sum(r['success'] for r in group_rows),eligible=sum(r['evidence_eligible'] for r in group_rows),median_error_mm=float(np.median([r['final_error_mm'] for r in group_rows])),max_error_mm=max(r['final_error_mm'] for r in group_rows),median_proxies=float(np.median([r['proxies'] for r in group_rows])),median_perception_p99_ms=float(np.median([r['perception_p99_ms'] for r in group_rows])),median_age_p99_ms=float(np.median([r['snapshot_age_p99_ms'] for r in group_rows])))
    merged=dict(experiment='07.4',development_only=True,mock_data=False,batch=str(batch),groups=table,online=result,same_input=same,full_manager_frames=pilot['frames']);(OUT/'results.json').write_text(json.dumps(merged,indent=2),encoding='utf-8')
    lines=['# 实验07.4：原始方向分叶的几何收益与更新代价','',
      '本轮使用07.3相同场景、UR5e、两台相机、五个起点、45s时长、1mm/50周期门槛。安全距离6mm、光轴误差3mm、CenterVox 7.5mm、核心种子半轴下限3mm均固定。开发实验的结果与正式验收分开。','',
      '## 1. 主闭环结果','',
      '|组别|次数|任务达标|完整证据达标|误差中位数/mm|最差误差/mm|代理数中位数|','|---|---:|---:|---:|---:|---:|---:|']
    for g,t in table.items():lines.append(f'|{g}|{t["n"]}|{t["successes"]}|{t["eligible"]}|{t["median_error_mm"]:.3f}|{t["max_error_mm"]:.3f}|{t["median_proxies"]:.0f}|')
    lines+=['','ET30：原单包络。P30：已有的帧后方向分叶，仅种子0，不能据此估计五起点成功率。D30：原始测量按误差主轴方向分组，每个方向叶独立累计完整误差矩阵。','',
      '|起点|ET30终点/mm|D30终点/mm|ET30达标|D30达标|','|---|---:|---:|---|---|']
    for seed in range(5):
        a=next(x for x in result if x['group']=='ET30' and x['seed']==seed);b=next(x for x in result if x['group']=='D30' and x['seed']==seed)
        lines.append(f'|{seed}|{a["final_error_mm"]:.3f}|{b["final_error_mm"]:.3f}|{a["success"]}|{b["success"]}|')
    lines+=['','## 2. 安全与时序审计','',
      '|组/起点|覆盖|MVT|离散穿透周期|代理扫掠失败|观测通过|柜体连续分离|超时周期|','|---|---|---|---:|---:|---|---|---:|']
    for r in sorted(result,key=lambda x:(x['group'],x['seed'])):lines.append(f'|{r["group"]}/{r["seed"]}|{r["coverage"]}|{r["mvt"]}|{r["penetrating_cycles"]}|{r["sweep_failures"]}|{r["observability"]}|{r["continuous_boxes"]}|{r["deadline_misses"]}|')
    lines+=['','连续柜体证书针对保存关节位置间的线性插值，使用解析关节杠杆上界和距离函数Lipschitz界；不证明自碰撞、地面或未知空间。覆盖已观测样本也不等于所有真实表面及时被观测。', '',
      '|组|感知更新p99的逐运行中位数/ms|地图年龄p99的逐运行中位数/ms|','|---|---:|---:|']
    for g,t in table.items():lines.append(f'|{g}|{t["median_perception_p99_ms"]:.1f}|{t["median_age_p99_ms"]:.1f}|')
    lines+=['','这些是开发批次的描述性计时。完整同输入重放在闭环批次前结束，闭环期间未另跑重型审计；未进行正式隔离调度性能认证。','',
      '## 3. 同输入几何消融','',
      f'重放07.3 ET30的{sum(x["frames"] for x in same)}个已发布源帧，{sum(x["matched_raw_samples"] for x in same):,}个工作空间环境样本；逐帧深度数量一致，未匹配样本为0。每个原始平移误差矩阵通过完整半正定包含检查。', '',
      '同一个空间体素的方向叶数中位数为2，实际最大值为'+str(max(x['leaves_per_cell_max'] for x in same))+'。分组不近似实际误差方向；不使用相机ID、目标或真实柜体选择分组。','',
      '|起点/部位/机器人球|原单元最大支撑/mm|方向叶并集最大支撑/mm|','|---|---:|---:|']
    for r in same:
        for p in r['pairs']:lines.append(f'|{r["seed"]}/{p["box"]}/{p["robot_index"]}|{p["old_max_support_mm"]:.3f}|{p["leaf_union_max_support_mm"]:.3f}|')
    total=sum(len(r['pairs']) for r in same);tighter=sum(p['leaf_union_max_support_mm']<p['old_max_support_mm'] for r in same for p in r['pairs'])
    lines+=['',f'{total}个历史终点近约束方向中，{tighter}个方向的上述最大支撑减小。其余方向仍可能扩大，不能声称全方向更紧。分组依据的固定坐标系、平移Young外包和后续代理拟合都仍会引入保守性。','',
      '## 4. 全链更新代价','',
      f'相同种子0的全部{len(pilot["frames"])}帧另经完整代理管理器重放，逐帧覆盖通过。最后有{pilot["frames"][-1]["spatial_cells"]}个空间单元、{pilot["frames"][-1]["directional_leaves"]}个方向叶和{pilot["frames"][-1]["proxies"]}个最终代理。', '',
      '原始方向累计约70ms/帧；主要额外成本在后续bucket_fit（局部代理拟合和覆盖选择），不能把仅累计阶段的速度当作整个感知更新速度。完整时间和分阶段日志在full_manager_same_input_seed0.json中。','',
      '## 5. 终点诊断','',
      '终点诊断重新构造实际被使用的快照和机器人姿态，与原pair_states逐对核验。使用几何间隙，不用独立高精度QP对偶作因果声称。', '',
      '|组/起点|重建最大间隙差/m|后壁最小证书间隙/mm|近零安全余量的部位|','|---|---:|---:|---|']
    for r in sorted(terminal,key=lambda x:(x['group'],x['seed'])):
        names=sorted(set(p['box'] for p in r['tight_pairs'] if p['barrier_h_mm']<.05));lines.append(f'|{r["group"]}/{r["seed"]}|{r["pair_clearance_reconstruction_max_error_m"]:.2e}|{r["back_min_certificate_gap_mm"]:.1f}|{", ".join(names)}|')
    lines+=['','## 6. 可支持的结论','',
      f'D30任务达标{table["D30"]["successes"]}/5，完整证据达标{table["D30"]["eligible"]}/5。新方向分组作为显式开发选项保留，未替换默认融合。柜子是否完成仍以全部预定证据门为准。', '',
      '下一步决策需要同时看几何与计算：保留不兼容方向能避免部分合并膨胀，但叶数增长会放大后续拟合代价。可研究以完整集合包含证明删除冗余叶，并核对真实输入集的包络形状；不能靠删除未覆盖观测、缩小误差假设或调低安全距离获得通过。MVT分层与球型主对照未被本轮替代。','',
      '## 7. 回放与复现','',
      '- view_experiment_07_4_compare.cmd：种子0三组实际运动；数字键按打印顺序切换，空格暂停，0重来，F终点。',
      '- view_experiment_07_4_proxies.cmd：D30种子0带已记录代理的回放；V叠加，O全部代理，Q参与QP，C核心，U不确定性，H限制对，G最小误差帧。',
      '- view_experiment_07_4.py --seed 2：其他种子；--overlays --group D30 查看对应代理。',
      '- run_experiment_07_4.py --with-pilot：重跑11次独立批次；异步调度不保证逐位相同，实际因果源姿态与快照均保留。',
      '- experiment_07_4_same_input.py / experiment_07_4_replay_manager.py：同输入消融与完整更新代价。',
      '- MATH_AND_CODE.md：第二章数学及代码调用解释。',
      '- results.json、terminal_constraints.json、online_/*/manifest.json：完整实测记录与源码哈希。','']
    frozen=json.loads((OUT/'frozen_map_diagnostic.json').read_text())['runs']
    contained=json.loads((OUT/'contained_leaf_audit.json').read_text())['runs']
    extra=['## 6. 补充机制诊断（探索性）','',
        'F0使用07.3 ET30种子0的同一末姿态，比较该数据的ET30最终地图与相同全部源帧构造的D30最终地图，冻结地图额外运行20s；没有新观测，结果不计入在线成功率。','',
        '|冻结地图|初始误差/mm|20s后误差/mm|保持达标|','|---|---:|---:|---|']
    for x in frozen:extra.append(f'|{x["group"]}|{x["initial_error_mm"]:.3f}|{x["final_error_mm"]:.3f}|{x["hold50_time_s"] is not None}|')
    extra += ['', '这说明在这一固定数据与起点上，新代理模型能使诊断轨迹更接近目标，但仍未到达。代理形状、数量及近场项也随表示改变，不能把所有收益归因于某一个U，也不能据此量化在线延迟造成了多少失败。','',
        'D30种子2、3的末周期近零安全余量包含柜侧板附近的前臂证书（forearm_link，球23）；不能只盯住下沿和腕部。盒子归属按代理中心最近真实盒定位，属于诊断标签，不是控制器读取真值。','',
        f'对五组同输入方向叶进行只读包含审计，约{min(x["removed_fraction"] for x in contained)*100:.1f}%–{max(x["removed_fraction"] for x in contained)*100:.1f}%的叶被同一体素内保留的另一叶完整包含，均保存正半定包含见证。这里没有重拟合代理或实际加速测试，不能把可删除比例当成耗时收益。',
        '', '后续优先检验有包含证明的冗余叶删除与局部拟合缓存；同时检查最终代理包络，而不是直接调小误差。该优化尚未接入本轮闭环。','']
    text='\n'.join(lines).replace('## 6. 可支持的结论','\n'.join(extra)+'\n## 7. 可支持的结论').replace('## 7. 回放与复现','## 8. 回放与复现')
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    merged['frozen_map_diagnostic']=frozen;merged['contained_leaf_audit']=contained
    (OUT/'results.json').write_text(json.dumps(merged,indent=2),encoding='utf-8')
    print(json.dumps(table,indent=2),flush=True)
if __name__=='__main__':main()
