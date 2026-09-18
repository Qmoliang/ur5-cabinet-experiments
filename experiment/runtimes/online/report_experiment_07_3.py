"""Aggregate only completed real 07.3 runs and produce a reviewable report."""
from pathlib import Path
import json,hashlib
import numpy as np
import mujoco
from model import set_configuration
from experiment_07_3_geometry import OUT,ROOT,render
from audit_experiment_07_3_continuous import audit

def main():
    batch=sorted((OUT/'online_').glob('*/manifest.json'))[-1];manifest=json.loads(batch.read_text(encoding='utf-8'));rows=manifest['results']
    assert len(rows)==10 and len({(r['group'],r['seed']) for r in rows})==10,'Wait for all ten runs'
    exact=json.loads((OUT/'exact_geometry.json').read_text())['runs'];same=json.loads((OUT/'same_input_join_fusion.json').read_text())['runs'];negative=json.loads((OUT/'same_input_fusion.json').read_text())['runs']
    result=[]
    for row in rows:
        folder=Path(row['summary']).parent;s=json.loads(Path(row['summary']).read_text());continuous=audit(folder)
        result.append(dict(group=row['group'],seed=row['seed'],final_error_mm=s['final_error_m']*1000,success=s['success'],evidence_eligible=s['evidence_eligible'],coverage=s['all_centervox_coverage_checks_passed'],mvt=s['all_mvt_oracle_checks_passed'],penetrating_cycles=s['exact_penetrating_cycles'],sweep_failures=s['sweep_audit_failures'],observability=s['observability_passed'],late_samples=s['observability_late_samples'],proxies=s['final_proxy_count'],deadline_misses=s['deadline_misses'],controller_p99_ms=s['controller_ms_p99'],perception_p99_ms=s['perception_pipeline_ms_p99'],snapshot_age_p99_ms=s['snapshot_age_ms_p99'],continuous_boxes=continuous['all_environment_box_segments_certified'],run=str(folder)))
        if row['seed']==0:
            m=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));d=mujoco.MjData(m);set_configuration(m,d,np.load(folder/'q_history.npy')[-1]);render(m,d,OUT/f'{row["group"]}_seed0_final.png')
    summary=dict(batch=str(batch),development_only=True,mock_data=False,exact_geometry=exact,online=result,same_input=same,rejected_support_box=negative)
    (OUT/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# 实验 07.3 结果：柜子可到达，但本轮融合修改仍未完成感知任务','',
        '本轮已执行真实几何诊断 5 次、双相机闭环 10 次、同源观测融合对照 5 组。数据均为实际运行；不覆盖 07.2。柜子阶段仍未验收。','',
        '## 1. 场景与不变条件','',
        '保留柜内尺寸、起点、目标及 65 个机器人证书球；四腿落地、亮地面，恢复官方 UR5e 视觉网格。柜体采用作者 shelves1.sdf 板件模板按既有抽屉尺寸适配，并改为不透明显示；不是原版 4Shelves，也不是原版 2IIWAs 桌架。新增两条前支撑，所有障碍表面参与相机射线与物理碰撞。视觉网格仅显示，不改变相机自遮挡使用的碰撞几何。','',
        '50 Hz、45 s、1 mm/50 周期、安全距离 6 mm、光轴误差 3 mm、CenterVox 7.5 mm、几何核心半轴种子下限 3 mm 全部固定。ET30=原融合，J30=直接累计原始平移误差矩阵。两个组真实闭环轨迹不同；同输入因素由单独相机帧回放隔离。','',
        '## 2. 真实几何诊断','',
        '五个起点全部达到并保持任务目标，第一次保持完成时间为 '+f'{min(r["first_hold50_s"] for r in exact):.2f}–{max(r["first_hold50_s"] for r in exact):.2f} s。'+
        '复用原 QP，只替换障碍几何和平面输入；近场项按真实盒数量计数且不做代理剪枝，所以它是任务可行性诊断，不是纯粹的 U 消融。','',
        '五条轨迹离散物理穿透均为 0；解析关节位移界证明所有关节插值段的机器人证书与柜体盒子保持分离。该连续证书不涵盖自碰撞和地面，且不证明整段严格维持 6 mm。离散最小证书间隙约 5.998 mm，存在微米量级求解容差。','',
        '## 3. 双相机完整闭环','',
        '|起点|ET30 终点误差/mm|J30 终点误差/mm|ET30 达标|J30 达标|','|---|---:|---:|---|---|']
    for seed in range(5):
        a=next(r for r in result if r['group']=='ET30' and r['seed']==seed);b=next(r for r in result if r['group']=='J30' and r['seed']==seed)
        lines.append(f'|{seed}|{a["final_error_mm"]:.3f}|{b["final_error_mm"]:.3f}|{a["success"]}|{b["success"]}|')
    lines+=['','|组别|达标次数|误差中位数/mm|误差最大值/mm|','|---|---:|---:|---:|']
    for group in ['ET30','J30']:
        group_rows=[r for r in result if r['group']==group];err=[r['final_error_mm'] for r in group_rows];lines.append(f'|{group}|{sum(r["success"] for r in group_rows)}/5|{np.median(err):.3f}|{max(err):.3f}|')
    lines+=['','两组均未完成任务。J30 有些起点更近，有些更差，不能作为已经解决问题的新版本推广。当前结果也不能与 07.2 的旧场景直接作同场景性能排名。','',
        '## 4. 审计与限制','',
        '|组/起点|覆盖|MVT|离散穿透周期|代理扫掠失败|观测及时|柜体连续分离|超时周期|','|---|---|---|---:|---:|---|---|---:|']
    for r in sorted(result,key=lambda x:(x['group'],x['seed'])):
        lines.append(f'|{r["group"]}/{r["seed"]}|{r["coverage"]}|{r["mvt"]}|{r["penetrating_cycles"]}|{r["sweep_failures"]}|{r["observability"]}|{r["continuous_boxes"]}|{r["deadline_misses"]}|')
    lines+=['','本轮时序数据属于开发机器上的描述性记录，运行期间有轻量分析工作，未做隔离调度性能复核。不能据此声称实时性改进。不存在达到正式 evidence_eligible 门槛的运行。','',
        '## 5. 同输入融合为什么没有一举解决','',
        f'重放 {sum(r["frames"] for r in same)} 个源帧，匹配 {sum(r["matched_samples"] for r in same):,} 个环境样本。原相机逐帧深度数量全部匹配；J30 对每个原始平移误差矩阵检查完整半正定包含关系，最小差矩阵特征值为正。',
        '', '第一次尝试 I30（原始支撑盒单次外包）覆盖正确，但单元体积比的逐种子中位数约为 '+f'{min(r["median_volume_ratio"] for r in negative):.2f}–{max(r["median_volume_ratio"] for r in negative):.2f} 倍，因 sqrt(3) 外扩而更保守；保留负结果，不进入主闭环对照。',
        '', 'J30 固定首帧坐标系直接累计原始误差矩阵的对角占优上界，减少部分下沿方向的膨胀；但另一台相机的观测方向变化会使固定坐标系的顶板方向上界增大。仅减少外包层数并不保证最终形状更紧。','',
        '|历史起点/部位/机器人球|旧单元最大支撑/mm|J30 最大支撑/mm|','|---|---:|---:|']
    for r in same:
        for pair in r['pairs']:lines.append(f'|{r["seed"]}/{pair["box"]}/{pair["robot_index"]}|{pair["old_cell_max_mm"]:.3f}|{pair["new_cell_max_mm"]:.3f}|')
    lines+=['','上表使用同一批已分配单元与同一方向，仅说明这些方向的变化；全方向安全来自矩阵包含检查。它不是两组闭环终点形状一一对应表。','',
        '## 6. 结论与下一步','',
        '本轮完成了可行性分离诊断、两个融合候选的检验与固定场景双组对照，但没有完成柜子任务。保留原方法为默认，不把 J30 自动替换为推荐版本。下一步需要处理双相机方向不兼容时的包络构造，并把误差集合的覆盖和通道保守性同时验证；不通过缩小传感器误差、安全距离或放宽成功门槛得到成功。MVT 与球型主对照仍待柜子方案稳定后进行。','',
        '## 7. 怎么看与复现','',
        '- 项目根目录 view_experiment_07_3_compare.cmd：种子 0 双组 MuJoCo 回放。1/2 切换组，空格暂停，0 重来，F 最后一帧。切换组会重新打开对应模型窗口。',
        '- view_experiment_07_3_exact.cmd：真实盒子诊断轨迹。',
        '- view_experiment_07_3.py --seed 2：可查看其他种子，全部失败样本保留。',
        '- MATH_AND_CODE.md：从相机到 QP 的数据流，以及第二章公式对应。',
        '- run_experiment_07_3.py：新建独立批次并运行双组五起点；原始时序快照保留，异步闭环不保证逐位重复。',
        '- experiment_07_3_same_input.py join：重放历史观测，复核同输入几何变化。',
        '- results.json：逐运行原始路径及审计字段；online_ 下每批 source/ 保留运行源码与哈希。','']
    terminal=json.loads((OUT/'terminal_constraints.json').read_text(encoding='utf-8'))['runs']
    assert len(terminal)==10
    gaps=[r['back_min_certificate_gap_mm'] for r in terminal]
    reconstruction=max(r['pair_clearance_reconstruction_max_error_m'] for r in terminal)
    assert all(r['collision_rows_by_box'].get('drawer_back',0)==0 for r in terminal)
    text='\n'.join(lines)
    note=("## 6. 终点约束核对\n\n"
          f"十次末周期几何重建与原 pair_states 日志的最大间隙差为 {reconstruction:.3g} m。近零安全余量集中在柜口下沿与前臂、顶板与腕部，个别行涉及相机外壳证书。后壁最小证书间隙 {min(gaps):.1f}–{max(gaps):.1f} mm，全部末周期均无后壁碰撞行。\n\n"
          "高精度独立反事实 QP 有未收敛情况；本轮不报告删除约束后的速度，也不把近零几何间隙称为对偶变量证明的唯一因果。terminal_constraints.json 保留无对偶的几何重建。\n\n"
          "## 7. 结论与下一步")
    text=text.replace('## 6. 结论与下一步',note).replace('## 7. 怎么看与复现','## 8. 怎么看与复现')
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(runs=len(result),successes=sum(r['success'] for r in result),coverage_all=all(r['coverage'] for r in result),mvt_all=all(r['mvt'] for r in result),penetrating_cycles=sum(r['penetrating_cycles'] for r in result),continuous_boxes_all=all(r['continuous_boxes'] for r in result),report=str(OUT/'REPORT.md')),indent=2))
if __name__=='__main__':main()
