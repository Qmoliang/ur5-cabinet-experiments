from pathlib import Path
import json,numpy as np
root=Path.cwd();out=root/'formal_results/experiment_07/terminal_diagnosis'
m=json.loads((out/'mvt_comparison.json').read_text(encoding='utf-8'))
u=json.loads((out/'uncertainty_decomposition.json').read_text(encoding='utf-8'))
assert len(m['cases'])==25
assert all(all(c['missing']==0 and c['extra']==0 for c in x['checks'].values()) for x in m['cases'])
assert all(x['raw_frame_counts_validated']==x['source_pose_count']==43 for x in u['runs'])
text=['# 实验 07：官方资产、MVT 实测与不确定性来源','',
'## 官方资产已经导入','',
'导入作者 wernerpe/iris_benchmarks 的 2IIWAs 桌架与 4Shelves 环境资产，生成保留现有 UR5e 的两个 MJCF 和预览。源目录树 SHA 为 c0cd495f38fc521350eba5c76f7354309c66ea21。UR5 关节、局部部件姿态、质量、惯性及全部 11 个原碰撞几何核对通过；只恢复原视觉网格。',
'两个环境文件位于 assets/experiment_07_official。其 README 和 import_manifest.json 记录地面调亮、整体落地等适配，以及作者源模型中视觉后板没有碰撞体等区别。它们属于资产预览，尚未接入新的正式柜子闭环试验。','',
'## MVT 的主要限制来自查询尺寸，而非椭球厚薄','',
'原生实现将代理放到满足 h_l > R_query,max + max(AABB半边长) 的最细可用层。每个代理只存一个中心格，每层查询中心格及 26 个邻格。',
'当前最大机器人证书球半径 61.85 mm，查询余量 40+6+0.005 mm，总公共查询半径 107.85 mm。层宽依次为 15、30、60、120、240 mm。120 mm 层只允许 AABB 最大半边长约 12.15 mm 以下的代理，因而大部分被分到 240 mm 粗层。先前将最后一层称为“最细层”的说法有误。',
'椭球的短轴小，不代表最长 AABB 半边长也小。把表面拟合得更薄，并不能保证层级分布更分散。也不能为了填满多层而强行拆分或扩大几何代理。','',
'## 同代理、同查询的实测','',
'五个种子各取 100、300、800、1500、2249 周期，共 25 个冻结状态，每种方法交错测量 20 次。表中为各状态中位耗时的中位数，每次调用查询全部 65 个机器人球。属于本机宽阶段微基准，不是端到端控制周期或正式统计置信区间。','',
'| 方法 | 查询耗时 / ms |','|---|---:|']
labels={'multilevel_simd':'当前多层 SIMD','multilevel_scalar':'多层标量','single_level_simd':'相同原生实现的单层 SIMD','radius_binned':'按机器人半径复制三套表','numpy_full_AABB':'NumPy 全量 AABB 扫描','tight_multilevel_simd':'使用 Q+U 精确 AABB 的多层 SIMD'}
for k,label in labels.items():text.append(f"| {label} | {m['summary_median_ms'][k]:.5f} |")
ratio=np.median([1-c['tight_candidate_pairs']/max(c['candidate_pairs'],1) for c in m['cases']])
text+=['',
'全部 25 个状态的索引候选与相应独立全量 AABB 判据一致，无漏项、无额外项。精确 Q+U AABB 使用 sqrt(Q_kk)+sqrt(U_kk)+offset；原方法使用单个外包椭球的对角项平方根。两者分别与自己的 AABB oracle 对比，不能要求不同包围盒的候选完全相同。',
f'精确 Q+U AABB 相比原外包椭球 AABB 的候选减少比例中位数为 {ratio*100:.2f}%；此批数据的查询耗时改善很小。',
'当前单层 SIMD 比多层略快，三套分半径表更慢；现有数据证明空间筛选和 SIMD 有用，但没有证明层数增加本身产生收益。NumPy 全量基线与原生实现存在语言和布局差异，不将全部比值归功于 MVT 算法。','',
'## 如何让多级结构发挥作用','',
'1. 先按覆盖精度构造自然尺度的代理：平整连续区域可较大，边缘和狭窄通道较小。避免为了展示多级而改变安全几何。',
'2. 可实验按障碍物本身尺寸选层，查询时依据当前机器人球半径和该层最大代理半径计算实际格子范围，以解除全局最大机器人球对所有代理的共同抬层。它将失去每层固定 27 格的性质，必须重新证明并验证不漏检，未必一定更快。',
'3. 保留一代理一引用、SIMD 连续数据布局和安全的格子包围盒预筛；针对真正需要变化的区域更新索引。',
'4. 用完整任务分布比较单层、多层、全量 AABB；记录构建/更新开销、内存、候选数、真实距离调用数和最终周期 p95/p99。自然尺寸范围有限时接受单层可能更优。',
'5. 不以“三层都有人”作为性能成功标准，也不缩小 near 阈值来制造层级利用率。','',
'## 不确定性包络的实测分解','',
'按正式配置重放五个种子各 43 个已发布相机源位姿：160×90、stride=1、最小量程 0.07 m、光轴深度误差上界 0.003 m、双相机遮挡自过滤、原生射线后端。全部 215 次重放的原始点数与冻结日志一致。对各限制代理所属体素，所有最终分配体素均在重放原始观测中匹配到。',
'下表均沿末态实际限制法向计算，不是三维最大半轴；原始测量项已经包含像素覆盖与深度误差的椭球外包。','',
'| seed / 代理 | 原始测量支撑最大值 / mm | 测量+体素代表点位移所需对称界 / mm | 已存体素 U 最大支撑 / mm | 最终代理 U 支撑 / mm |','|---|---:|---:|---:|---:|']
for run in u['runs']:
    for p in run['pairs']:
        text.append(f"| {run['seed']} / {p['proxy_id']} | {p['raw_measurement_support_max_mm']:.3f} | {p['raw_measurement_plus_cell_relocation_bound_mm']:.3f} | {p['stored_cell_U_max_support_mm']:.3f} | {p['proxy_U_support_mm']:.3f} |")
text+=['',
'体素内不同点的空间差异、把不对称平移集合对称化、Minkowski 外包以及跨帧/跨方向的 Loewner 上界合并，都可能增加支持范围。最终代理 U 与已存体素 U 很接近，说明显著增量在体素/历史聚合阶段已经出现，单独修改末级聚类并不能消除它。',
'代码中的 1.05 限制比较的是某次合并的矩阵迹与最大成员矩阵迹，并不是“最终每个方向最多膨胀 5%”，更不约束全流程相对原始观测的累计膨胀。',
'因此，目前“包络有明显额外保守量”的判断有数据支持；但某个方向更紧不等于已经构造出全方向覆盖有效的替代椭球。也不能将 11 mm 直接改为 4 mm 或在没有独立性假设时用 sqrt(N) 缩小界。','',
'## 建议的修改顺序','',
'先给原始相机误差、体素代表点位移、跨帧合并、聚类合并分别建立预算和覆盖校验；减少对已经外包的椭球反复再外包。可保持稳定方向的原始支撑区间摘要，在发布时构造一次证书，或按视向/局部曲面分组保留多个有效子包络。固定方向摘要不能未经证明替代全方向覆盖。',
'优先在柜口下沿和顶板验证更紧证书能否仍覆盖原测量集合，之后做保持原安全距离和成功阈值的闭环对照，再评估 MVT 的尺度和耗时变化。此轮没有改变正式控制器、深度误差假设、6 mm 安全余量或 1 mm 成功阈值。','',
'数据：mvt_comparison.json、uncertainty_decomposition.json；复现脚本：benchmark_et30_mvt_use.py、audit_et30_uncertainty_components.py。']
(out/'official_assets_mvt_uncertainty_review.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps({'report':str(out/'official_assets_mvt_uncertainty_review.md'),'tight_AABB_candidate_reduction_median_percent':ratio*100,'validated_camera_frames':215,'mvt_cases':25}))
