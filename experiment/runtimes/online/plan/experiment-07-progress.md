# 实验 07.2 执行进度（2026-09-16）

## 当前状态

- E7-D0：passed。协议、追踪矩阵和表图契约已经建立。
- E7-D1：passed after correction。抽屉的 8 个历史 box、`q0` 和目标与 4.3
  数值相同，只增加两只后侧落地腿。鸟笼前平面已修正到 `x=0.350 m`；两场景
  五个冻结初态均零接触，目标分别有 1 和 3 组无接触 IK。
- E7-D2：passed smoke。厚度下限只作用于核心 `Q_i`，`U_i` 和
  `delta_i` 保持分离；相关 3 个单测通过。
- E7-D3a：passed。S/SA/E0/ET30 在每个场景的 generation-0 CenterVox 数组
  逐字节相等、SHA-256 相同，全部直接核心覆盖通过。完整轨迹交叉回放待正式前导。
- E7-D4：passed for local continuous cross-section。抽屉中心截面上 SA 有 4.244 mm 开口见证，ET30 有
  10.518 mm 严格开球，最近点最大 KKT 残差 `8.34e-9`；该证据不等于完整
  关节空间路径证书，也不支持“球一定封死”的旧叙事；闭环成功仍由 D6 判定。
- E7-D5：候选 recall 和抽屉轨迹逻辑控制输入等价通过；但 ET30 在五次正式末态的 120 mm 层占比都低于 5%，地图增长后层级结构门失败。
- E7-D6：20 次正式运行已完成并聚合；任务成功 2/20，完整证据资格 0/20，柜子尚未做完。
- E7-D7：按协议顺序暂不开始，先处理 7A 的收敛、sweep 和可观测性失败。
- E7-D8：未开始；只有鸟笼 B1 通过后才加入夹爪和物体。

## 冻结主组

| Group | 含义 | 角色 |
|---|---|---|
| S | cap-irredundant sphere | 少量大球的负载基线 |
| SA | adaptive-irredundant sphere | 球表示与 MVT 层级主对照 |
| E0 | 原始 PCA 表面椭球 | 零厚度基线 |
| ET30 | `a_min >= 3.00 mm` 椭球 | 非极薄主椭球组 |

3.75 mm 与 7.50 mm 作为上界消融保留。3.00 mm 的选择只用了覆盖、核心轴比、
连续截面净空和 MVT 层级结构，没有使用正式任务成功率。

## 07.2 初始快照

| Scene | Group | proxies | axis ratio p95 | MVT 120/240 mm | oracle |
|---|---:|---:|---:|---|---|
| drawer | S | 208 | n/a | 5 / 203 | pass |
| drawer | SA | 2937 | 1.00 | 768 / 2169 | pass |
| drawer | E0 | 1550 | 411.19 | 550 / 1000 | pass |
| drawer | ET30 | 1530 | 13.81 | 393 / 1137 | pass |
| birdcage | S | 216 | n/a | 1 / 215 | pass |
| birdcage | SA | 3283 | 1.00 | 339 / 2944 | pass |
| birdcage | E0 | 1963 | 384.21 | 389 / 1574 | pass |
| birdcage | ET30 | 1921 | 13.41 | 203 / 1718 | pass |

前三个 15/30/60 mm 层仍为空，这是当前全局最大机器人半径加
`NEAR+safety` 的层选择结果。初始快照只主张 120/240 mm 两层被有效使用，不声称
五层都被用满。

## 300 周期开发结果

| Scene | Group | final error | 说明 |
|---|---:|---:|---|
| drawer | S | 350.9 mm | 推进很弱 |
| drawer | SA | 179.0 mm | 球多尺度对照 |
| drawer | E0 | 109.2 mm | 原始薄椭球 |
| drawer | ET30 | 96.3 mm | 当前抽屉最好 |
| birdcage 07.2 | ET30 | 393.3 mm | 仅小幅推进，B1 尚未完成 |

这些运行不使用正式 50 Hz wall-clock pacing，也没有达到 `<1 mm × 50 cycles`，
所以不能计入成功率。

## 7A 正式结果（50 Hz，5 seeds/group）

| Group | task success | evidence eligible | final error median | proxy median | deadline misses | penetration |
|---|---:|---:|---:|---:|---:|---:|
| S | 0/5 | 0/5 | 342.75 mm | 325 | 0 | 0 |
| SA | 0/5 | 0/5 | 283.43 mm | 4157 | 2 | 0 |
| E0 | 2/5 | 0/5 | 1.93 mm | 2538 | 0 | 0 |
| ET30 | 0/5 | 0/5 | 4.20 mm | 2490 | 0 | 0 |

E0 的任务成功率为 40%，Wilson 95% CI 为 11.8%--76.9%；样本仅 5 次。其
seed 0 虽达到目标，但有 1 次 continuous sweep audit failure；seed 3 达到目标，
但有 21 个 late-observability 样本。因此两次均不满足完整 evidence gate。四组
共 20 次源码哈希完全一致，覆盖和 MVT oracle 全通过，MuJoCo 穿透周期均为 0。

ET30 把轴比 p95 从约 410 降到约 14，但严格成功从 E0 的 2/5 降为 0/5，显示
3 mm 厚度消耗了深抽屉最后几毫米余量。它的五次正式末态又有约 94.6%--95.4%
代理落入 240 mm 层，未保持预注册的层级结构门。SA 说明多小球可以形成尺度层级，
但约 4,000 个代理仍未换来深插入成功，并出现两次 deadline miss。

结果文件：

- `tables/experiment_07/T27_drawer_formal_runs_07_2.csv`
- `formal_results/experiment_07/drawer_formal_summary_07_2.json`
- `formal_results/experiment_07/drawer_failure_diagnosis_07_2.json`
- `figures/experiment_07/F14_drawer_formal_outcome_07_2.png`

## 已保存的失败证据

1. 抽屉整面落地支撑使窄通道 IK 搜索失败：
   `scene_gate_attempt_00_full_side_supports_failed.json`。
2. 随机 IK 未命中历史窄通道分支：
   `scene_gate_attempt_01_random_ik_branch_missed.json`。
3. 07.0 鸟笼 `Q0_V3` 与前栏杆穿透：
   `initial_seed_gate_attempt_00_front_plane_collision.json`。该场景的旧鸟笼
   烟雾/300 周期目录不进入统计。
4. 原始 ET7.5 几乎全部落在 240 mm 层；按半径复制三份索引的原型虽改善层级
   分布，但查询慢约 1.9–2.4 倍、内存引用为 3 倍，因此不接入控制器。
5. 旧 paired-certificate 要求“球闭、椭球开”，而实验 07 的 SA 已有开口见证；
   其 `paired_certificate_passed=false` 是叙事不适用，不是最近点核失败。

## 核心证据

- `formal_results/experiment_07/static_scene_gate/scene_gate.json`
- `formal_results/experiment_07/initial_seed_gate.json`
- `formal_results/experiment_07/same_cloud_initial_snapshot_audit.json`
- `formal_results/experiment_07/mvt_selected_group_audit_07_2.json`
- `formal_results/experiment_07/drawer/paired_cross_section_sa_et30_initial_smoke/certificate.json`
- `tables/experiment_07/T25_same_cloud_initial_smoke.csv`
- `tables/experiment_07/T26_mvt_level_smoke_07_2.csv`
- `tables/experiment_07/T27_drawer_formal_runs_07_2.csv`
- `formal_results/experiment_07/drawer_formal_summary_07_2.json`
- `formal_results/experiment_07/drawer_failure_diagnosis_07_2.json`

## 下一步

07.3 只针对三项已定位问题设计：入口唇边采用有物理依据的局部厚度策略；让左壁风险区在 cycle 102 前发布；将连续 sweep guard 从 audit 证据转为可验证的执行约束。先写 07.3 修订协议并重过静态/同点云门；7A 未达到稳健证据门前不进入 7B B1/B2。

