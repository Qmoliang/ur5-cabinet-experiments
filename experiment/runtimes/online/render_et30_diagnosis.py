"""Render the terminal diagnosis from measured frozen-run artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import mujoco
from diagnose_et30_terminal import ROOT, OUT
from formal_protocol_v3_viewer import RaggedSnapshots
from model import build_robot_certificate, certificate_world_state, set_configuration
from run_protocol_v3_async_online import _protocol_scene

report=json.loads((OUT/'terminal_constraints.json').read_text(encoding='utf-8'))
continuation=json.loads((OUT/'frozen_continuation.json').read_text(encoding='utf-8'))
r=report['runs'][0]
run=Path(r['source_run'])
summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
scene=_protocol_scene(summary.get('camera_scene_version',summary['scene_version']))
model=mujoco.MjModel.from_xml_path(str(run/'scene.xml'))
data=mujoco.MjData(model)
set_configuration(model,data,np.load(run/'q_history.npy')[-2])
robot=build_robot_certificate(model)
positions,_,radii=certificate_world_state(model,data,robot)
_,a=RaggedSnapshots(run/'causal_proxy_snapshots.npz').at_cycle(2249)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(1,2,figsize=(14,5.3),layout='constrained')
ax=axes[0]
for box in scene.boxes:
    if box.name not in ('drawer_bottom','drawer_ceiling','drawer_back'): continue
    x,y,z=box.center; sx,sy,sz=box.half_size
    ax.add_patch(Rectangle(((x-sx)*1000,(z-sz)*1000),2*sx*1000,2*sz*1000,color='#8198a6',alpha=.7))
for i,(p,radius) in enumerate(zip(positions,radii)):
    if p[0]<.23: continue
    ax.add_patch(Circle((p[0]*1000,p[2]*1000),radius*1000,facecolor='#36a3c2',edgecolor='#21748b',alpha=.10))
pair=r['tight_pairs'][0]
ri=pair['robot_index']; oi=int(np.flatnonzero(a['proxy_ids']==pair['proxy_id'])[0]); c=a['centers'][oi]
n=np.column_stack([np.cos(np.linspace(0,2*np.pi,361)),np.zeros(361),np.sin(np.linspace(0,2*np.pi,361))])
Q=a['ellipsoid_shapes'][oi]; U=a['proxy_uncertainty_shapes'][oi]
q=n@Q/np.sqrt(np.einsum('ij,jk,ik->i',n,Q,n))[:,None]
u=n@U/np.maximum(np.sqrt(np.einsum('ij,jk,ik->i',n,U,n)),1e-15)[:,None]
for vertices,color,label in [(c+q+u,'#9e6ccc','核心 + 不确定性'),(c+q,'#e9a336','核心椭球')]:
    ax.add_patch(Polygon(vertices[:,[0,2]]*1000,facecolor=color,edgecolor=color,alpha=.6,label=label))
ax.add_patch(Circle(positions[ri,[0,2]]*1000,radii[ri]*1000,facecolor='none',edgecolor='#cf3868',lw=2,label='限制前臂的证书球 #35'))
ax.scatter([630],[580],marker='*',color='#229947',s=100,label='目标')
ax.annotate('柜口下沿：实际限制位置',xy=(350,510),xytext=(450,478),arrowprops={'arrowstyle':'->'},ha='center')
ax.text(789,578,'后壁',rotation=90,ha='center',va='center')
ax.set(xlim=(255,825),ylim=(460,695),xlabel='深入柜子的方向 x / mm',ylabel='高度 z / mm',title='seed 0：末态几何侧向投影')
ax.set_aspect('equal'); ax.legend(loc='upper left',fontsize=8)
ax.text(.02,.02,'此图为 x-z 投影；距离数值使用完整三维计算。',transform=ax.transAxes,fontsize=8)
ax=axes[1]
colors=['#0072b2','#d55e00','#009e73','#cc79a7','#7065aa']
for item in continuation['runs']:
    if item['variant']!='unchanged_controller': continue
    x=[0]+[t['extra_time_s'] for t in item['trace']]
    y=[item['initial_error_mm']]+[t['error_mm'] for t in item['trace']]
    ax.plot(x,y,color=colors[item['seed']],lw=2,label=f"seed {item['seed']}")
ax.axhline(1,color='#333333',linestyle='--',lw=1,label='1 mm 阈值')
ax.set(xlabel='在原 45 秒之后额外续跑的仿真时间 / s',ylabel='末端位置误差 / mm',title='冻结最后地图，保持原控制参数续跑')
ax.grid(alpha=.2); ax.legend(ncol=3,fontsize=9)
fig.suptitle('ET30 末态诊断：柜口下沿与顶板约束，后壁未参与末态 QP',fontsize=15)
figure=ROOT/'figures/experiment_07/F15_ET30_terminal_diagnosis.png'
fig.savefig(figure,dpi=160)
plt.close(fig)

base={x['seed']:x for x in continuation['runs'] if x['variant']=='unchanged_controller'}
lines=['# ET30 最后几毫米为什么到不了：冻结记录诊断','',
'本记录基于实验 07.2 的五次 ET30 正式运行。新增分析单独保存，没有修改场景、目标、控制器或正式数据。','',
'## 结论','',
'末态限制来自柜口底板对应的前臂约束，部分种子还叠加顶板对应的手腕约束。后壁没有进入末态 QP。NEAR 惩罚降低了运动速度，但不是所有失败的充分解释。3 mm 核心半轴下限不能被单独认定为失败根因。','',
'## 方法与可复核性','',
'读取每次最后一周期的实际因果代理快照；q_history 保存执行后的姿态，因此使用倒数第二个姿态重建最后一次 QP。末端执行后位置与日志完全一致；五次分别重建 466、455、440、412、418 个碰撞约束，全部匹配记录，最大间隙重建误差小于 6e-17 m。',
'独立以 1e-9 OSQP 容差重解同一个 QP，检查约束对偶值与速度松弛，再按来源板件移除硬约束做瞬时消融。代理来源按其中心到实体板件表面的最近距离归类，仅用于事后分析。','',
'## 1. 排除后壁','',
'五次末态的机器人证书球到后壁的最小三维间隙为 107.74–114.02 mm。五次末态后壁碰撞约束数均为 0，因此移除后壁约束完全不改变解。目标 x=0.630 m，后壁内表面 x=0.800 m，两者相隔 170 mm。','',
'## 2. 真正起作用的约束','',
'| seed | 正式最终误差 / mm | 正对偶值的硬约束对应部件 | 保持原控制器额外续跑 | 续跑终点误差 / mm |',
'|---|---:|---|---|---:|']
for x in report['runs']:
    pairs=[p for p in x['tight_pairs'] if p['dual']>1e-5]
    text='；'.join(f"{p['body']} #{p['robot_index']} → {p['box']}" for p in pairs)
    b=base[x['seed']]
    lines.append(f"| {x['seed']} | {x['final_error_mm']:.4f} | {text} | {b['extra_time_s']:.2f} s | {b['final_error_mm']:.4f} |")
lines += ['',
'底板限制球 #35 属于 forearm_link；seed 0 中它的中心 x≈0.323 m，尚在 x=0.350 m 的柜口之前。末端向前并不代表所有连杆仅作水平平移，关节联动会令前臂靠近柜口下沿。顶板同时限制手腕时，局部目标方向更难实现。','',
'## 3. 为什么核心椭球看着不厚，控制器却已到安全边界','',
'控制器实际使用 E(Q) ⊕ E(U) ⊕ B(offset)，并在机器人球半径以外再保留 6 mm 安全距离。C 显示的核心不等于完整避障体。3 mm 参数是核心短半轴的种子下限，不是完整厚度；包络缩放后实际半轴还会变大。','',
'以 seed 0 的前臂 #35 与底板代理 #141 为例：','',
f"- 核心半轴：{', '.join(f'{v:.3f}' for v in pair['core_semiaxes_mm'])} mm。",
f"- 不确定性椭球半轴：{', '.join(f'{v:.3f}' for v in pair['uncertainty_semiaxes_mm'])} mm。",
f"- 在实际限制法向上的核心支撑半径：{pair['core_support_along_normal_mm']:.3f} mm；不确定性额外支撑：{pair['uncertainty_support_along_normal_mm']:.3f} mm。",
f"- MuJoCo 前臂源碰撞几何到实体底板：{pair['source_geom_to_box_gap_mm']:.3f} mm。",
f"- 机器人证书球到实体底板：{pair['certificate_to_box_gap_mm']:.3f} mm。",
f"- 机器人证书球到 Q+U 代理：{pair['surface_gap_mm']:.6f} mm。",
'- 扣掉 6 mm 安全余量后，h≈0。实体没有碰撞，但代理约束已经生效。',
'这些距离的最近点可能不同，不能把差值解释成同一直线上的精确厚度分解；它们说明保守代理比实体更早限制运动。不确定性包络不能在未经覆盖校验的情况下直接删除。','',
'## 4. 因果对照与时间因素','',
'在 seed 0 同一末态，误差下降速度为 0.0529 mm/s；移除底板硬约束后为 0.4201 mm/s，移除后壁为 0.0529 mm/s；保留全部硬安全约束、仅去掉 NEAR 速度惩罚，瞬时速度为 1.0607 mm/s。这些是离线瞬时对照，不是可执行安全方案。',
'seed 1、2 同时受到上下硬约束，去掉 NEAR 惩罚也未解决续跑停滞。seed 3 保持原参数续跑 13.92 s 后满足位置误差 <1 mm 且连续 50 周期；说明该次存在明显时间与收敛速度因素。其正式 45 s 结果仍为失败，原观测审计问题也不会被续跑消除。',
'冻结地图续跑没有新增相机观测，也没有重新做连续扫掠和观测及时性审计。它只用于区分机制，不能升级为正式成功证据，且不能证明不存在另一条可行路径或姿态。','',
'## 5. 场景外观核实','',
'experiment_07_scenes.py 确实只在后侧添加 cabinet_ground_left / cabinet_ground_right 两根支撑，以保持 v4.3 内部几何不变。这是本地简化模型，并未导入官方演示素材。model.py 的 floor 材质仍为深灰 (0.28, 0.30, 0.34)。两条腿和昏暗地面来自实现选择，不是实验方法的必要条件，也没有满足用户要求的完整落地柜视觉效果。',
'用户提供的官方参考链接本次网页读取失败，因此本记录不声称已核对该页面资产文件或可复用许可。外观整改应采用明亮地面与完整支撑结构；本轮诊断保留冻结场景。','',
'## 后续应查什么','',
'优先分解柜口下沿和顶板的观测不确定性、聚类包络与机器人证书保守量，进行同点云、同姿态下的覆盖有效性对照。不要根据目前结果直接缩小 6 mm 安全余量、删除不确定性包络或放宽成功阈值。随后评估保持有效安全证书时能否通过姿态调整解除局部上下约束。','',
'## 文件','',
'- diagnose_et30_terminal.py：原记录重建、三维距离、对偶值与瞬时消融。',
'- continue_et30_terminal.py：冻结地图续跑。',
'- formal_results/experiment_07/terminal_diagnosis/terminal_constraints.json：五次约束证据。',
'- formal_results/experiment_07/terminal_diagnosis/frozen_continuation.json：续跑轨迹采样。',
'- figures/experiment_07/F15_ET30_terminal_diagnosis.png：侧视投影和续跑曲线。',
'',
'距离校验备注：seed 2 的后壁 mj_geomDistance 原始返回出现与保守证书下界不一致的 0 值，该值已标记并排除。后壁结论使用证书球—实体盒解析距离与 QP 约束记录，不依赖该异常值。']
path=OUT/'ET30_terminal_diagnosis.md'
path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'figure':str(figure),'report':str(path),'core_support_mm':pair['core_support_along_normal_mm'],'uncertainty_support_mm':pair['uncertainty_support_along_normal_mm']},ensure_ascii=False))
