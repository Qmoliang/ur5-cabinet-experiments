# 07.4 原始方向叶：从观测到控制

第二章 2.7 使用 c+E(Q)+E(U) 作为障碍证书。07.3 的 J30 仍要求一个空间单元输出一个 U，方向不一致时会填满原本没有观测误差的区域。

本轮 D30 在同一个空间体素中保存若干 U_l，表示并集 union_l(c+E(U_l))。它仍是一个共享地图；不能把两台相机同一表面的测量做交集，因为这些测量不一定对应同一个未知物理点。

## 改变的位置

run_experiment_07_4.py → run_protocol_v3_async_online.PerceptionWorker → incremental_proxy_manager.update → raw_directional_uncertainty.RawDirectionalJoin。

原始点 p 与完整误差矩阵 U_raw 进入方向分组。分组使用 U_raw 的主轴，不读取机器人目标、不读取真实柜体，也不依赖相机名称。平移 d=p-c 后，先用原来解析 Young 外包得到 B，保证 p+E(U_raw) 包含于 c+E(B)。每个方向叶固定自己的坐标系并累计 B 的完整矩阵上界。

后续 pointcloud_proxy.py 接收每个叶作为一个需要覆盖的单元证书，构造核心 Q 和代理误差 U。覆盖必须包括所有叶，不能只挑方便机器人通过的方向。MVT、最近支撑法向、删冗余平面、QP 和关节积分仍用现有公式。

## 三个组回答三个问题

- ET30：一个空间体素最终一个误差包络。
- P30：已有分叶方法，先把本帧所有测量合起来，再决定跨帧是否分叶。
- D30：先把不同方向的原始测量分开，再各自累计历史；避免同帧交叉观测先变厚。

## 正确性与代价

方向格只决定谁与谁共享一个外包，绝不会把实际法向替换为格子中心。每个原始误差矩阵都执行全矩阵半正定包含检查；独立测试还检验旋转、平移后的实际椭球边界点。

更多叶并不必然意味着更好的实验：后续拟合、覆盖选择、MVT和QP可能更贵；地图发布延迟可能导致机器人尚未看到关键障碍就已经接近。因此本轮既记录几何支撑变化，也记录完整更新耗时、地图年龄与观测及时性。只看“包络薄了”不能下结论。

## 阅读顺序

1. plan/experiment-07-4-protocol.md：固定任务、对照、证据门。
2. experiment_07_4_same_input.py：同一批观测几何对照。
3. raw_directional_uncertainty.py：本轮数学核心。
4. incremental_proxy_manager.py：如何把叶接入原有覆盖链。
5. experiment_07_4_replay_manager.py：完整拟合代价。
6. run_experiment_07_4.py：真实双相机闭环。
7. 本目录 REPORT.md：完整结果及限制。
