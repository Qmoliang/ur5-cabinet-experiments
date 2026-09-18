# 正式协议 v3 结果任务包

## 结果结论

同一 `formal_drawer_two_camera` 场景、同一 UR5 机器人球链、同一最终目标和相同控制参数下，球型在线 LiuQP 在 45 s 内未到达，椭球型在线 LiuQP 于 12.26 s 首次进入 18 mm 成功区并保持 10 个周期。两个控制器均未使用人工路径、RRT、路点、随机扰动、真值碰撞回退或在线 sweep 缩放。

球型失败由独立连续截面证书支持：最大净空的连续上界为 -5.827 mm。椭球在同一截面的全机器人球—环境代理最小余量为 +5.034 mm。该结果支持“椭球缓解球证书各向同性外扩造成的假闭塞”，不支持一般非凸局部极小的全局完备性结论。

## 主结果

| 指标 | 球型 LiuQP | 椭球型 LiuQP |
|---|---:|---:|
| 在线周期/仿真时长 | 2250 / 45.00 s | 623 / 12.46 s |
| 成功 | 否 | 是 |
| 首次进入 18 mm | — | 12.26 s |
| 最终误差 | 355.705 mm | 17.148 mm |
| 最大末端 x | 0.275287 m | 0.613447 m |
| QP 求解率 | 100% | 100% |
| 精确 MuJoCo 穿透周期 | 0 | 0 |
| 控制计算 p99 | 8.716 ms | 16.355 ms |
| 计算超过 20 ms 的周期 | 2 | 2 |
| 相机/代理发布快照 | 25 | 13 |

## 硬审计

- 静态几何：`formal_results/final_two_camera/static_geometry_gate_final/static_geometry_gate.json`。
- 在线球/椭球：`formal_results/final_two_camera/formal_evidence_final_v5/`。
- 加速严格等价：`formal_results/final_two_camera/known_packed_equivalence_final_v3/acceleration_equivalence_report.json`。
- MVT/SIMD 微基准：`formal_results/final_two_camera/final_mvt_simd_benchmark_v5.json`。
- 综合机器验证：`formal_results/final_two_camera/formal_protocol_v3_evidence_manifest_v5.json`。

三态地图重放、暴力 AABB 候选重建和连续 proxy sweep 重放均在控制完成后执行；所有报告都声明 `online_control_or_publication_timing_affected=false`。球 sweep 的最小代理余量为负，来自已存在的保守代理重叠；审计按 CONTACT/RECOVERY 的“不加深、单调恢复”语义判定，0 个周期被标记为 unsafe。该值不能写成“全程代理净空非负”。

## 加速结论

在 200 个冻结代理上，5 层 MVT-Scalar 相对 O(N) AABB oracle 的 controller p99 加速约为球 1.025 倍、椭球 1.028 倍。MVT-AVX2 的完整控制 p99 受小批量开销和操作系统噪声影响，没有快于标量；独立原生查询微基准在正式约 500 代理负载下中位数快 1.38%，在 15480 个 CenterVox 点压力输入下快 6.64%。论文应据此表述为“真实 AVX2 已实现并等价，小规模整体加速有限”，不能写成显著 SIMD 加速。

## 已生成的论文证据产物

- `figures/formal_v5/fig1_online_comparison.png/.svg`：球/椭球全周期主对照。
- `figures/formal_v5/fig2_centervox_sensitivity.png/.svg`：5/7.5/10 mm CenterVox 敏感性。
- `tables/formal_v5/`：在线主结果、严格加速等价、CenterVox 敏感性三张表。
- `figures/formal_v5/videos/`：球型受阻与椭球型进入抽屉的因果 MuJoCo 回放。

所有产物均由真实 v5 日志生成，并有 SHA-256 清单；未使用手填论文数据。视频不重跑控制器，只显示保存的机器人状态、当时已观测点云和当时已发布代理。

## 尚未通过的研究目标

CenterVox 分辨率/代理数量敏感性已完成。0.3 mm 有界噪声探索暴露了保守证书膨胀与地图边界重放问题：椭球组没有到达，连续 sweep 与地图重放没有全部通过，因此尚不能形成多种子噪声鲁棒性正结论，也不能宣布整个研究全部完成。修订不确定性/在线代理合并属于方法变化，必须先更新并冻结协议，再进行多随机种子重跑；不得删除或隐藏该负面结果。
