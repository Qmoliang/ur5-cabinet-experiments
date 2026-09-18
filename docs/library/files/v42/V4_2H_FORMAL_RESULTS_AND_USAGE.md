# v4.2h 正式结果、观看与复现

## 1. 最终结论

最终有效对照不是旧 v1–v12，也不是 v4.2e 的 7.5 mm 候选，而是 v4.2h：

- 同一个原始 140 mm 抽屉场景、同一 q0、同一唯一目标 `(0.63, 0.40633091, 0.58)`。
- 两台随机械臂运动的局部深度相机；地图从 unknown 开始，逐帧更新 free/occupied/unknown。
- 两组都使用 12.5 mm CenterVox、同一观测输入规则、同一 2×2 切向分片和同一代理中心/数量规则。
- 球半径严格等于匹配椭球最长半轴；椭球使用真实最近点，不存在 `ellipsoid-fast`。
- 椭球最终 `U=0`、offset=0；6 mm 只属于 LiuQP 硬安全约束，没有重新膨胀核心椭球。
- 无 RRT、人工路点、随机扰动、真值碰撞回退、预载全局地图或成功后提前终止。

| 指标 | 球型 LiuQP | 椭球型 LiuQP |
|---|---:|---:|
| 运行 | 3000 周期 / 60 s | 3000 周期 / 60 s |
| 最终误差 | 318.914 mm | 0.970 mm |
| 严格 `<1 mm` 连续 50 周期 | 失败 | 成功，58.18 s 完成 |
| 控制器 p99 | 11.41 ms | 13.49 ms |
| 穿透周期 | 0 | 0 |
| 资格 | 合格负对照 | 合格正结果 |

同一球型因果快照上的连续证明给出：匹配球截面最大净空上界 \(-8.586\ \mu m\)，严格闭塞；把同中心、同数量代理换回匹配椭球后，中心开口为 6.288 mm。成功椭球轨迹自身快照开口为 7.575 mm。

## 2. 直接观看

在 PowerShell 中运行：

```powershell
cd D:\MuJoCo\liu_qp_reproduction\ur5_liuqp_iris_scenes
D:\anaconda3\envs\simple\python.exe view_strict_goal_v4_2h.py
```

常用快捷键：

- `1` / `2`：切换球型 / 椭球型并从头播放。
- `Space`：暂停或继续；`N`：单步；`0`：回到开头；`F`：末帧；`G`：最小误差帧。
- `V`：关闭或恢复所有算法叠加层。UR5 本体、相机外壳和抽屉物理几何始终保留。
- `P`：当时已经发布的因果点云；`O`：全部障碍代理。
- `B`：MVT/AABB 候选代理；`Q`：ordered erase-remove 后真正进入 QP 的代理。
- `M`：free/occupied 地图；未出现在稀疏表中的有界体素就是 unknown。
- `R`：机器人 65 个球证书；只隐藏证书球，不隐藏机械臂本体。
- `H`：当前限制配对；`C`：QP 核心椭球；`E`：外包显示层；`U`：不确定性层。
- `-` / `=`：减速 / 加速；`L`：循环播放。

v4.2h 中 `U=0` 且外包层与核心层相同，所以默认只显示 `C`，避免过去看到的“两层椭球/大球套小球”视觉重复。

只检查文件和因果一致性、不打开窗口：

```powershell
D:\anaconda3\envs\simple\python.exe view_strict_goal_v4_2h.py --check
```

## 3. 重新运行

下面命令必须使用新的 `--label`；脚本拒绝覆盖已有证据目录。

```powershell
D:\anaconda3\envs\simple\python.exe run_strict_goal_v4_2h_cv12p5_sphere.py --label user_reproduction --affinity-layout isolated_12_2_18

D:\anaconda3\envs\simple\python.exe run_strict_goal_v4_2h_cv12p5_ellipsoid.py --label user_reproduction --affinity-layout isolated_12_2_18 --ellipsoid-pair-threads 6
```

快速 24 秒检查可增加 `--smoke-cycles 1200`，但 smoke 结果不能替代 60 秒正式证据。

## 4. 事后重放

对每个新生成的内层运行目录依次执行：

```powershell
D:\anaconda3\envs\simple\python.exe reconstruct_candidate_pairs.py <run_dir>
D:\anaconda3\envs\simple\python.exe compress_candidate_proxy_ids.py <run_dir>
D:\anaconda3\envs\simple\python.exe reconstruct_proxy_sweep_audit.py <run_dir>
D:\anaconda3\envs\simple\python.exe verify_recorded_causal_occupancy.py <run_dir>
```

正式配对截面证书：

```powershell
D:\anaconda3\envs\simple\python.exe certify_formal_paired_cross_section.py <sphere_root> <ellipsoid_root> <new_output_dir> --grid-size 301
```

## 5. 证据位置

- 球型正式根目录：`formal_results/final_two_camera/formal_strict_goal_v4_2h_matched_sphere_camera_quarter_affinity_isolated_12_2_18_formal_paired`
- 椭球型正式根目录：`formal_results/final_two_camera/formal_strict_goal_v4_2h_reachable_partitioned_thin_camera_quarter_affinity_isolated_12_2_18_formal_paired`
- 最终同快照截面证书：`formal_results/final_two_camera/formal_strict_goal_v4_2h_paired_cross_section_same_snapshot_formal_paired/certificate.json`

完整候选 CSV 用于审计；viewer 读取 `candidate_proxy_ids.npz`，避免加载数十百万重复的机器人球—代理对。
