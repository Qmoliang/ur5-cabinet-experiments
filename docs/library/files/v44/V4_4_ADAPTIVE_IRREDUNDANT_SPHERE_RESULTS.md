# v4.4 自适应不可约球证书：设计、文献口径与当前结果

## 1. 冻结边界

本轮不修改 v4.3 椭球 LiuQP 的代理几何、最近点算法、QP 三状态、相机、控制参数或冻结结果。椭球正式结果仍为最终误差 0.177 mm、控制器 p99 18.992 ms、最终 2,187 个代理；其完整因果观测旧审计仍保留为 48 个 late、3 个 never，不追溯修改。

只修改球型地图代理发布层。机器人证书、两台移动深度相机、增量地图、CenterVox、MVT/AABB/SIMD、LiuQP 三状态和目标定义均不变。

## 2. 相关工作如何处理遮挡

| Source | 已核对事实 | 对本实验的含义 |
|---|---|---|
| [RMMI, IROS 2025 / arXiv:2408.16206](https://arxiv.org/abs/2408.16206) | 实机场景被假定为静态；每个场景先由机器人采集 30 对带位姿深度图，再用 iSDF 重建，之后执行 reaching。论文没有声称在每次 reaching 中持续解决永久盲区。 | RMMI 通过任务前重建规避了在线永久盲区问题，不能用来要求本实验在第 0 周期看到整个柜体。 |
| [iSDF, RSS 2022](https://www.roboticsproceedings.org/rss18/p012.pdf) | iSDF 本身支持从移动相机的连续带位姿深度流增量训练 SDF。 | 流式更新是地图能力，但“未知区域安全”仍需由控制/审计协议定义；神经场的合理补全不等于可证明 FREE。 |
| [Flacco and De Luca, RA-L 2017, DOI:10.1109/LRA.2016.2535859](https://iris.uniroma1.it/handle/11573/887612) | 使用两台 Kinect，离线初始化与相机布置有关的 depth-grid，在线快速融合多传感器深度信息。 | 两台互补视角深度相机是已有实验采用的合理配置；固定工作空间覆盖和在线融合可以分层处理。 |

因此，本研究保留两台腕部/前臂相机。可观测性应区分：

1. `never`：任务相关风险表面直到结束仍未被任何相机观测；这是视角/永久盲区问题。
2. `late`：最终能看到，但地图发布时间晚于该表面首次进入风险带；这是处理延迟/发布频率问题。

最新球型正式运行的 `never=0`、`late=8`，说明两台相机已经消除了该轨迹上的永久盲区；剩余问题是地图约 0.274 Hz、快照年龄 p95 7.64 s，而不是需要第三台相机。该失败原样保留，不能改写为完整观测门通过。

## 3. 球证书数学定义

对第 \(i\) 个 CenterVox 单元，保留代表点 \(p_i\)、方向测量不确定性 \(U_i\) 和标量残差 \(\delta_i\)。球型证书采用保守但解析的各向同性支持

\[
s_i=\sqrt{\lambda_{\max}(U_i)}+\delta_i.
\]

候选球 \(B(c_j,r_j)\) 完整覆盖该单元的充分条件为

\[
\lVert p_i-c_j\rVert+s_i\le r_j.
\]

仅最小化球数量会退化为少量大球。为避免目标相关调参，候选球的额外几何外包预算固定为一个 CenterVox 对角线：

\[
\epsilon_{\rm geom}=\sqrt{3}\,h_{\rm cv},
\qquad
r_j\le \max_{i\in C_j}s_i+\epsilon_{\rm geom},
\]

同时仍满足全局尺度上限 70 mm。若局部簇不满足该式，按最大点云展宽轴确定性二分，直到每个子簇满足；这不使用目标、人工路径、真值障碍标签或当前机器人位置。

在固定候选集合上执行：

1. 确定性 lazy-greedy set cover；
2. 反向删除所有仍可删除的已选球；
3. 将每个保留球缩到其实际负责单元所需的最大 \(\lVert p_i-c_j\rVert+s_i\)。

最终证书是“确定性、覆盖安全、inclusion-minimal 的近似最少球证书”，不声称求解了 NP-hard 全局最少集合覆盖。

LiuQP 原有 ordered erase-remove 仍只负责当前机器人球/当前控制周期的 QP 行删减，和这里的地图证书删减是两个不同层次。

## 4. 三次修正为何必须保留

| 球型版本 | 最终发布球 | 最终半径范围 | 最终误差 | 控制 p99 | sweep 失败 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| v4.3 统一 70 mm | 3,256 | 70 mm | 377.940 mm | 10.044 ms | 0 | 覆盖安全，但不是最少/不可约证书。 |
| v4.4a 直接以 70 mm 做 set cover | 175 | 38.8–70.0 mm | 356.073 mm | 10.832 ms | 2,462 | 数量最少目标偏向大球；新发布球会虚假覆盖机器人，拒绝。 |
| v4.4b 局部拟合半径、无几何误差上限 | 573 | 5.6–68.9 mm | 329.894 mm | 10.426 ms | 1,716 | 有局部自适应，但仍允许少数过大球，拒绝。 |
| v4.4c CenterVox 分辨率约束 | 3,697 | 5.3–28.4 mm | 263.724 mm | 14.030 ms | 0 | 覆盖、不可约、MVT、半径、连续 sweep、零穿透和控制实时门通过；未达到目标。 |

v4.4c 的最终候选数为 6,680，发布数为 3,697，反向删除 42 个已选冗余球。最终误差 263.724 mm，严格小于 1 mm 的任务门未通过。它没有被人为要求失败；失败是运行结果。

v4.4c 可观测性为 `late=8, never=0`，所以完整因果感知门未通过。地图管线 p50/p99 为 3.494/4.588 s，有效发布频率 0.274 Hz；控制器 p99 为 14.030 ms，仍通过 20 ms 控制门。必须分别报告低频地图成本和高频 QP 成本。

## 5. 当前可以支持的结论

1. 原 3,256 个统一大球不是最少证书，不能继续作为最终球型基线。
2. 单纯追求最少数量会选出少量大球并制造虚假 CONTACT；必须同时约束几何外包误差。
3. 将球的额外外包误差限制到 CenterVox 分辨率后，需要约 3,697 个小球，连续安全审计恢复通过，目标方向进展明显优于统一大球，但仍未达到任务点。
4. 冻结椭球版以 2,187 个代理达到 0.177 mm；球型若保持相近的窄向保真度，需要更多代理。这支持“椭球在表示效率上更优”的结论。
5. 当前数据不支持“椭球每个控制周期一定更快”：椭球精确最近点窄阶段更贵。可支持的说法是椭球用更少代理保留狭窄通道，而 MVT/AABB/SIMD 使两种表示的 50 Hz 控制仍可实时。

## 6. 结果位置

- 冻结椭球：`formal_results/final_two_camera/formal_strict_goal_v4_3_surface_core_ellipsoid_camera_quarter_cv7.5mm_cr70mm_u1p05_stride1_vmax180mmps_affinity_perception_heavy_formal_redesign_no_coreinflate_r70_paired`
- v4.4a：`formal_results/final_two_camera/formal_strict_goal_v4_3_surface_core_sphere_camera_quarter_cv7.5mm_cr70mm_u1p05_stride1_vmax180mmps_affinity_perception_heavy_adaptive_cover_formal1_adaptive_irredundant_sphere_cover`
- v4.4b：`formal_results/final_two_camera/formal_strict_goal_v4_3_surface_core_sphere_camera_quarter_cv7.5mm_cr70mm_u1p05_stride1_vmax180mmps_affinity_perception_heavy_adaptive_localradius_formal2_adaptive_irredundant_sphere_cover`
- v4.4c：`formal_results/final_two_camera/formal_strict_goal_v4_3_surface_core_sphere_camera_quarter_cv7.5mm_cr70mm_u1p05_stride1_vmax180mmps_affinity_perception_heavy_resolution_bounded_formal3_adaptive_irredundant_sphere_cover`

