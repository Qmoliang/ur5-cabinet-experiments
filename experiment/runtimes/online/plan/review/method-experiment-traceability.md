# 方法—实验可追溯矩阵（冻结基线有限补证；v5 重复路线已停止）

| Contribution / requirement | Method module | Experiment / audit | Table / figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| v4.4 球型与 v4.3 椭球型永久冻结 | `freeze_v4_4_v4_3_formal_baselines.py` | `verify_v4_4_v4_3_frozen_baselines.py` 逐文件 SHA-256 复核 | T11 | 两组是不可改写的正式历史基线 | 已冻结并复核 36 文件 |
| 新场景输入/源码/候选可精确恢复 | recording contract R0--R7 | manifest、源码/DLL 快照、完整 `(p,U,delta)`、候选 CSR 覆盖矩阵复核 | T22, T23 | 只对记录门全部通过的运行作可复现或同输入因果主张 | 协议已锁定；代码实现待用户审核后处理 |
| 两相机先融合地图、禁止双套代理 | depth camera + occupancy + proxy manager | 冻结 summary/source 与因果快照核验 | T17, F5 | 代理来自一个融合地图而非按相机重复发布 | 冻结实现已满足；不是新增贡献 |
| 同点云而非同代理数量 | crossed causal snapshot replay | 冻结最终 CenterVox 哈希审计 | T12, F2 | 同一 CenterVox 下比较表示效率；数量可不同 | **未通过**：14930/14504，哈希不同；逐点 scalar offset 不完整，不以当前 v5 冒充旧实现 |
| CenterVox 不丢证书 | CenterVox source-ID/`U_i`/`delta_i` | 原始 occupied → cell → proxy 三层覆盖 | T12, T13, F6 | 稀疏化仍覆盖全部 observed occupied 数据 | 旧版通过；v5 待重跑 |
| 给定尺度球不可约覆盖 | sphere candidate family + set cover | 30/50/70/resolution-bounded 每快照审计 | T13, F3 | 相对冻结候选族 inclusion-minimal，不声称全局最少 | v4.4c 单组通过；v5 多尺度待实现 |
| 给定最长尺度椭球不可约覆盖 | ellipsoid candidate family + support oracle | 同上 | T13, F3 | 椭球也不得以密集重叠代替不可约覆盖 | 待实现 |
| 无重复中心/完全包含无见证代理 | proxy reduction audit | 唯一中心、包含对、唯一见证单元 | T13 | 每个发布代理均不可单独删除 | 球 v4.4c 已检查；v5 两表示待实现 |
| 精确椭球最近点与分离平面 | safeguarded Newton + bisection + warm start | 数值残差、稠密表面 oracle、逐周期统计 | T14, F4 | LiuQP 使用真实最近点法平面，无 ellipsoid-fast | 旧版通过；v5 待统一复核 |
| LiuQP 三状态一致 | controller state machine | 边界单测、逐 pair/row 日志 | T15, F4 | NORMAL/NEAR/CONTACT 与 soft/recovery 规则相同 | 待正式审计 |
| 无 RRT/路点/扰动/真值回退 | static source/config audit | 禁止符号、命令哈希、轨迹来源审计 | T14 | 成功只来自 final-goal-only LiuQP | 待 v5 门槛 |
| 真实目标可达 | MuJoCo exact goal IK + obstacle-free run | 静态目标审计、无障碍同控制器 | T14, F1 | 场景真实几何与控制能力不构成伪失败 | 旧版通过；v5 配置待冻结 |
| 球闭塞/椭球开口连续证明 | Lipschitz cross-section certifier | 冻结最终快照与第 168 周期截面 | T14, F3 | 仅在连续证书通过的截面声称闭塞 | **全截面未通过**：局部固定 `x,y` 线闭合；同一 `x` 全截面有 +0.265 mm 开口，正面最终快照有 +2.248 mm 开口 |
| 停滞归因 | limiting-pair logger/viewer | 第 168 周期 `c,h,state` 与后续状态统计 | T15, F4 | 冻结球运行进入持久局部硬 QP 死锁 | 已通过：`c=3.916 mm`,`h=-2.084 mm`，随后 2832 周期零速度 |
| 在线局部感知 | wrist + forearm depth cameras | observed-before-risk / never / late | T17, F5 | 不要求初始全图；要求接近风险面前已发布 | 当前基线均未完整通过；v5 待解决 |
| free/occupied/unknown 因果地图 | incremental occupancy map | 录制数据重放与版本单调审计 | T17, F5 | 不读未来帧、不把 unknown 当 free | 旧版通过；v5 待重放 |
| C++ 多级 MVT/27 邻域 | native MVT | 当前内核对冻结两组全部 proxy generation 回放 | T16, F7 | 当前候选查询相对 full-scan 无漏检 | 已通过 7800 次 scalar/AVX2 行对照；历史二进制未重跑 |
| AABB/SIMD 等价 | native broad phase | full-scan/scalar/AVX2 候选 ID | T16, F7 | 当前宽阶段加速不改变候选集合 | 两表示均 100% 一致；当前 C++/DLL 哈希与历史冻结不一致 |
| 端到端实时 | async perception + proxy + controller | 模块 p50/p95/p99、年龄和 deadline | T16, T17, F7 | 50 Hz 控制与冻结地图年龄门同时通过 | 当前基线有 deadline/地图频率缺口 |
| 主任务案例对照 | frozen online sphere/ellipsoid LiuQP | v4.3/v4.4 各一次固定时长闭环 | T15, F8 | 同系统设置下椭球运行成功、球运行失败；不作普遍 Pareto 或同点云因果外推 | 描述性案例成立；n=1/组，不做显著性检验 |
| 鲁棒性 | deterministic perturb/noise/drop/resolution suite | 预注册 formal seeds | T18, F9 | 结论不依赖一个起点或一帧噪声 | 待运行 |
| 控制周期 Viewer | formal causal viewer | 读取不可变 snapshot ID | F1, F5, F6 | 可视对象与控制使用的数据一致，UR5 本体不可隐藏 | 基线可看；v5 待新入口 |

没有对应实验或明确 limitation 的贡献不得进入论文主结论。失败、超时、不可解、
不可观测和 deadline miss 运行全部保留并计入分母。


## 实验 07 追加追踪矩阵

| Contribution / requirement | Method module | Experiment / audit | Table / figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| 4.3 抽屉内部几何保持、柜体落地 | `experiment_07_scenes.py` + scene geometry verifier | 旧/新内部 box 数值逐项比较、地面 AABB 与目标 IK | T24, F11 | 新视觉场景保持原狭长插入任务且实体落地 | E7-D1 passed |
| 表面椭球与厚度先验分离 | `pointcloud_proxy.py` minimum core semi-axis | `t_min={0,3.0,3.75,7.5} mm` 同快照消融、覆盖/轴比审计 | T25, F12 | 最小厚度对形状、覆盖和通道的影响；不预设优越 | ET30 frozen；同快照覆盖 passed |
| 球/E0/ET 同点云公平比较 | experiment 07 crossed replay | source/CenterVox SHA-256、候选族、覆盖矩阵复核 | T25, F12 | 只有哈希相同的行支持表示层因果比较 | S/SA/E0/ET30 generation-0 passed；full crossed replay pending |
| 多级 MVT 确实使用多层 | native MVT level assignment/export | level dump、full oracle、4×19,565 轨迹查询逻辑行 hash | T26, F13 | SA/E0/ET30 两层各 >=5%、dominant <=90% 且无漏检时才声称层级有效 | initial passed；formal ET30 map-growth 0/5，层级主张失败 |
| 落地抽屉闭环主实验 | frozen experiment 07 runner | S/SA/E0/ET30 各 5 次 final-goal-only | T27, F14 | 本抽屉场景的系统级任务、安全、实时结果 | complete：task S 0/5, SA 0/5, E0 2/5, ET30 0/5；evidence eligible 0/20 |
| 大型鸟笼预抓取泛化 | grounded bar cage + unchanged controller | B1 各 5 次入口和预抓取 | T28, F11/F15 | 抽屉冻结设置对栏杆入口的场景泛化 | 待 E7-D7 |
| 鸟笼内抓取 | compact gripper + object phase log | B2 到达/闭合/抬升/保持 | T29, F15 | 只有分段证据完整才声称完成抓取 | 待 E7-D8 |





07.3 development extension: see experiment-07-3-traceability.md.


07.4 原始方向分组 → 同输入全矩阵包含/支撑对照 + ET30/D30五起点闭环；P30区分帧后与帧前分组。失败不作安全成功或泛化声称。
