# 正式协议 v5.0 阶段门

阶段严格顺序执行；失败结果永久保留。修改冻结变量必须升级协议并从 D0 重启。

| Gate | Required evidence | Pass rule | Current status |
|---|---|---|---|
| D0 协议与基线冻结 | v5.0 协议、追溯表、表/图契约；v4.4/v4.3 SHA-256 | 文档一致，旧结果零修改 | passed：36 个文件内容寻址冻结并由独立入口复核 |
| D1 候选族与不可约覆盖 | 两表示多尺度候选、覆盖 oracle、唯一见证 | 全 CenterVox 覆盖；zero-witness=0；duplicate/contained redundant=0 | pending |
| D2 同点云交叉回放 | 两条闭环流上每快照双表示构造 | source/CenterVox hash 成对相同；不读目标/机器人生成候选 | pending |
| D3 控制器与精确核 | 65 robot balls、三状态、soft/fallback、Newton/bisection/warm start、禁用项 | 单测和逐周期审计全部通过 | pending |
| D4 静态几何与连续截面 | target IK、无障碍 QP、球闭塞/椭球开口、限制对归因 | 连续界和安全裕量达到协议；无规划路径注入 | pending |
| D5 在线因果感知 | 两相机融合地图、版本重放、observed-before-risk | never=0；late-risk=0；地图/代理年龄通过 | pending |
| D6 查询等价与实时 | full oracle、MVT27、AABB、SIMD、完整模块 timing | recall=1；最终行/命令 hash 相同；control p99<=20 ms；零 deadline miss | pending |
| D7 正式闭环与鲁棒性 | 多尺度、至少5 formal seeds、固定时长、全部失败入表 | 椭球 `<1 mm × 50 cycles`；零穿透；形成同点云 Pareto 证据 | pending |
| D8 交付复核 | 原始日志、聚合、表图、Viewer、hash、负结果 | 一键 verifier 全通过且无 mock 数据 | pending |

D7 不要求球在所有尺度失败，也不要求球/椭球代理数量相同。若球缩小后成功，
其更多代理和计算成本是正式结果的一部分。


## 实验 07 阶段门

实验 07 使用独立编号，避免与历史 `protocol_drawer_v7` 相机布局混淆。完整定义见
`plan/experiment-07-protocol.md`。

| Gate | Required evidence | Pass rule | Current status |
|---|---|---|---|
| E7-D0 | 协议、追踪矩阵、表图契约 | 文档一致；历史冻结文件零修改 | passed 2026-09-16 |
| E7-D1 | 落地抽屉/鸟笼 XML、尺寸、截图、IK、五初态接触 | 7A 内部几何与 4.3 相同；两目标可达；五初态零接触 | passed after birdcage correction 2026-09-16 |
| E7-D2 | 厚度感知拟合、单测、覆盖/轴比审计 | 全覆盖；无不确定性重复；参数可恢复 | passed smoke 2026-09-16 |
| E7-D3 | S/SA/E0/ET30 同快照交叉回放 | 输入/CenterVox hash 相同 | generation-0 passed；full trajectory pending |
| E7-D4 | 连续通道与限制对审计 | 数值误差界完整；无路径注入 | passed local cross-section：ET30 10.518 mm 严格开球；任务成功由 D6 判定 |
| E7-D5 | 五层 MVT/full oracle | SA/E0/ET30 至少两层各占 >=5%；dominant <=90%；全组 recall=1；控制等价 | oracle/equivalence passed；formal ET30 structure failed 0/5 after map growth |
| E7-D6 | 7A 每组 5 次闭环 | 失败全保留；任务/安全/实时全报告 | executed/reported：20 runs；task 2/20，evidence-eligible 0/20，drawer outcome failed |
| E7-D7 | 7B B1 每组 5 次 | 冻结 7A 控制参数完成预抓取 | blocked by protocol sequence：7A not robustly complete |
| E7-D8 | 7B B2 每组 5 次 | 到达、闭合、抬升、保持证据完整 | pending |
| E7-D9 | 聚合、图、回放、manifest | verifier 通过；无 mock/手填 | pending |





