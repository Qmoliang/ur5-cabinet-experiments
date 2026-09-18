# v4.3 不增厚核心椭球正式对照

## 1. 本轮重新设计

本轮废弃“为了覆盖而把核心椭球短轴再次膨胀”的做法。在线代理严格拆成三部分：

\[
\mathcal E_{\rm core}
=\{c+u\mid u^\top Q_{\rm core}^{-1}u\le 1\},
\]

其中 \(Q_{\rm core}\) 只包络 CenterVox 保留下来的真实表面代表点。相机像素/深度不确定性仍为独立 \(U\)，CenterVox 单元残差仍为独立标量 \(r_{\rm cv}\)。椭球 LiuQP 的方向支持为

\[
\rho(n)=
\sqrt{n^\top Q_Rn}
+\sqrt{n^\top Q_{\rm core}n}
+\sqrt{n^\top Un}
+r_{\rm cv}+\delta_s .
\]

没有把 \(U\)、\(r_{\rm cv}\) 或安全余量重新融合进 \(Q_{\rm core}\)，运行记录中的 `direct_centervox_thin_axis_inflation=1.0`。

70 mm 是最终代理支持的**尺度上限**，不是核心短轴。超过该上限的表面片按空间/不确定性递归细分，每个子代理重新拟合并逐帧做覆盖审计。球型使用统一 70 mm 最终球证书；椭球型只要求最长最终支持不超过 70 mm，短轴仍由表面点直接决定。

正式椭球最终代理共 2,187 个。核心短轴最小值、中央值和 p99 分别为 0.100 mm、0.124 mm 和 0.141 mm；核心长轴 p99 为 57.623 mm。独立相机不确定性长轴 p99 为 16.963 mm，二者没有合并回核心。

## 2. 冻结验收标准

- 同一 MuJoCo 抽屉场景、q0、目标、两台移动 D405、增量 free/occupied/unknown 地图、CenterVox、MVT/AABB/SIMD、三状态 LiuQP 和求解参数。
- 禁止 RRT、人工路径/路点、随机扰动、姿态引导、真值回退和 sweep 命令修改。
- 3,000 个控制周期；成功必须严格小于 1 mm 并连续保持 50 周期，成功后不提前停止。
- CenterVox 覆盖、代理尺度、MVT oracle、连续 sweep、MuJoCo 零穿透和控制 p99 小于 20 ms 均为硬门。

## 3. 正式结果

| 指标 | 球型 LiuQP | 椭球型 LiuQP |
|---|---:|---:|
| 严格成功 | 否 | 是 |
| 首次稳定完成 | 未发生 | 36.04 s |
| 最小误差 | 377.940 mm | 0.177 mm |
| 最终误差 | 377.940 mm | 0.177 mm |
| 控制器 p99 | 10.044 ms | 18.992 ms |
| CenterVox / MVT / 尺度门 | 通过 | 通过 |
| 连续 sweep 失败 | 0 | 0 |
| sweep 修改命令 | 0 | 0 |
| MuJoCo 穿透周期 | 0 | 0 |
| 因果 observed-before-risk | 通过 | **未通过** |

因此，本轮已经通过“核心椭球不重新增厚且末端真实到达任务点”的设计验收，也得到安全、实时的球型失败负对照。它尚未通过完整在线感知证据门：椭球轨迹仍有 48 个风险表面样本发布过晚、3 个柜体左外缘样本从未发布到控制地图。正式结果必须表述为“控制/几何验收通过，完整因果感知协议未完成”，不得写成全协议证据合格。

`camera_compensated` 两相机候选完整保留，但在线椭球最终误差为 1.822 mm，未达到任务点，故已拒绝，不能用它替换正式场景。

## 4. 结果位置与核验

- 球型正式根目录：`formal_results/final_two_camera/formal_strict_goal_v4_3_surface_core_sphere_camera_quarter_cv7.5mm_cr70mm_u1p05_stride1_vmax180mmps_affinity_perception_heavy_formal_redesign_no_coreinflate_r70_paired`
- 椭球正式根目录：`formal_results/final_two_camera/formal_strict_goal_v4_3_surface_core_ellipsoid_camera_quarter_cv7.5mm_cr70mm_u1p05_stride1_vmax180mmps_affinity_perception_heavy_formal_redesign_no_coreinflate_r70_paired`

核验核心设计验收：

```powershell
D:\anaconda3\envs\simple\python.exe .\verify_v4_3_no_core_reinflation.py
```

要求包括相机因果观测在内的完整协议时：

```powershell
D:\anaconda3\envs\simple\python.exe .\verify_v4_3_no_core_reinflation.py --require-full-evidence
```

第二条当前应返回非零退出码，这一失败是预期且必须保留的剩余研究问题。
