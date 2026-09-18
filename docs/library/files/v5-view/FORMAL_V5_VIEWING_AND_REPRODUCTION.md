# 正式 v5：观看与复现

> 本文记录历史 v5 证据。当前严格 `<1 mm`、真实椭球最近点和同快照闭塞对照已经由 v4.2h 取代；请优先阅读 `V4_2H_FORMAL_RESULTS_AND_USAGE.md`，并使用 `view_strict_goal_v4_2h.py`。

## 直接观看已完成实验

- 球型 LiuQP 因果回放：`figures/formal_v5/videos/formal_v5_sphere_causal_replay.mp4`。
- 椭球型 LiuQP 因果回放：`figures/formal_v5/videos/formal_v5_ellipsoid_causal_replay.mp4`。
- 在线主结果图：`figures/formal_v5/fig1_online_comparison.png`。
- CenterVox 敏感性图：`figures/formal_v5/fig2_centervox_sensitivity.png`。

视频只读取正式 v5 保存的关节状态、当时已观测点云和当时已发布代理，没有重新运行控制器，也没有补入未来点云。

## 交互式 MuJoCo viewer

在 PowerShell 中运行：

```powershell
cd D:\MuJoCo\liu_qp_reproduction\ur5_liuqp_iris_scenes
D:\anaconda3\envs\simple\python.exe formal_protocol_v3_viewer.py
```

按键：

- `1` / `2`：球型 / 椭球型。
- `Space`：播放或暂停；`N`：单步；`0`：回到开头；`L`：循环。
- `V`：总叠加层开关。它不会隐藏 UR5、相机外壳、抽屉或书架本体。
- `P`：因果点云；`O`：全部代理；`B`：MVT/AABB 候选；`Q`：QP 激活代理。
- `M`：free/occupied/unknown 地图；`R`：机器人证书球。
- `C` / `E` / `U`：分别切换已知、观测和未知相关显示；`-` / `=` 调整播放速度。

只做数据一致性检查、不打开窗口：

```powershell
D:\anaconda3\envs\simple\python.exe formal_protocol_v3_viewer.py --check
```

## 核验冻结证据

```powershell
D:\anaconda3\envs\simple\python.exe verify_formal_protocol_v3_evidence.py
D:\anaconda3\envs\simple\python.exe -m unittest discover -p "test_*.py"
```

当前冻结证据的综合结果为 29/29 项通过、98/98 项测试通过。`hard_realtime_all_cycles_passed=false`，因此只能声明 p99 软实时，不能声明所有周期都满足 20 ms。

## 重新运行同参数主控制实验

下面的命令写入新目录，不覆盖正式 v5：

```powershell
D:\anaconda3\envs\simple\python.exe run_protocol_v3_async_online.py --representation both --index-mode mvt_simd --maximum-cycles 2250 --unknown-policy observed_only --realtime-pacing --sweep-guard-mode post_control_audit --perception-executor process --output formal_results\reproduction_user --centervox-size 0.0075 --maximum-uncertainty-union-inflation 1.25 --camera-width 320 --camera-height 180 --camera-pixel-stride 1 --camera-layout wrist_forearm --scene-version formal
```

这是单一最终目标 LiuQP；没有 RRT、路点、随机扰动、真值碰撞回退或在线 sweep 缩放。正式论文证据仍应引用 `formal_results/final_two_camera/formal_evidence_final_v5` 及其审计清单，除非新运行也完成候选、连续 sweep、三态地图和观测时限的全部事后重放审计。

## 当前结论边界

冻结无额外深度噪声主场景中，球型在 45 s 内失败，椭球型在 12.26 s 首次到达；连续截面证明球证书严格闭塞而椭球证书保留开口。0.3 mm 有界噪声探索组仍是负结果，尚不能据此声明多随机种子噪声鲁棒性。
