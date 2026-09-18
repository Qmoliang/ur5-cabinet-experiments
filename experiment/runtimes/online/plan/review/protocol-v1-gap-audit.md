# 协议 v1 初始差距审计

审计日期：2026-08-29。

## 可复用且已验证

- MuJoCo UR5e 模型、球证书生成、深度渲染、free/occupied/unknown 地图、CenterVox/在线代理管理的旧模块和 48 个既有测试。
- 受保护 Newton + 必要二分 + 初值乘子的椭球最近点求解器。
- 现有 C++ AVX2/SoA AABB 查询及标量 oracle 测试，可作为新 multilevel MVT 的底层内核。
- 旧 v1-v12 轨迹、图和耗时分析保留为历史工程原型。

## 已发现的正式协议偏差

- 旧 incremental 场景包含夹爪、腕部加肩部两相机；旧 shelf 场景包含长工具。
- 旧在线 runner 含 RRT/A*/IRIS region/pure-pursuit/task region，不能作为 final-goal-only LiuQP。
- 旧椭球控制器改变了机器人表示并加入 orientation/posture/dither 等项，不是单变量环境表示对照。
- 旧结果使用已知完整环境或仅极少控制步，不能证明单腕相机因果增量地图闭环。
- 现有 native 表主要是单层重叠体素 AABB 索引，不满足“真正多级、按层 27 邻域”的正式命名。
- 旧表、图和 stage gate 混合了 204 代理、969 球细化及不同协议结果，不能作为 v1 正式证据。

## 本轮已完成的纠偏

- 新增 `protocol_drawer_v1`：仅一个腕部 D405 尺寸相机外壳，无夹爪和长工具。
- 新增已知环境 matched-cell 球/椭球代理：共享 205 个稳定 ID/中心/局部单元并保守覆盖。
- 新增统一 `ProtocolLiuQPController`：共享机器人球，球解析距离或椭球真实最近点，完整三状态和原始 erase-remove 广义式；删除所有规划/路点/姿态/抖动接口。
- 新增协议不变量单测；正式实验规划文档已重写，旧结果不迁移。

## 尚未完成，禁止提前声称

- 独立连续精确路径与截面闭塞/开放证明。
- 已知环境完整 50 Hz 主对照及失败鲁棒性。
- 单腕深度的增量三态地图闭环、在线代理原子快照和覆盖审计。
- 真正 C++ multilevel 27-neighbor MVT 及完整 SIMD 窄前阶段。
- 四级在线消融、p99 实时门和正式控制周期 viewer。
