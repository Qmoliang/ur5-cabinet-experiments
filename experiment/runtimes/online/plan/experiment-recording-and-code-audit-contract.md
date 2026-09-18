# 在线闭环与同快照离线配对：记录和代码审核协议 v1

> 生效日期：2026-09-01。适用于此后所有 UR5 狭窄通道、抽屉、书架及其他
> 场景。它不追溯修改 v4.3/v4.4，也不把停止的 v5 诊断结果升级为正式证据。

## 1. 两类实验及各自回答的问题

每个新场景最多包含两类互补实验：

1. **在线闭环实验**：球型和椭球型 LiuQP 各自使用真实因果局部地图运行，回答
   “机器人实际是否到达、是否安全、是否实时、在哪里停滞”。两条轨迹和后续
   相机观测允许自然不同。
2. **同快照离线配对**：从真实在线运行中冻结一份完整 CenterVox 证书输入，
   分别构造球和椭球，回答“相同输入下表示、覆盖、截面和查询成本有什么差异”。
   该步骤不运动机器人，不修改闭环结果。

在线任务结果不能冒充同点云机制证据；离线配对也不能冒充机器人闭环成功。

## 2. 所有实验共同的不可变记录

每次运行或离线配对开始前必须创建唯一 `experiment_id/run_id`，随后保存：

- 场景 ID、表示、数据来源轨迹、正式/诊断标志、随机种子和 UTC 时间；
- 原始 `scene.xml`、初始 `q0`、唯一目标、控制周期、固定总周期和成功门；
- 相机内参、实体安装位姿、分辨率、请求帧率、有效距离和深度误差模型；
- CenterVox 尺寸、`U_i/delta_i` 定义、球半径或椭球最长轴限制；
- `d_safe`、NORMAL/NEAR/CONTACT 阈值、QP 权重、关节速度/加速度界和 OSQP 参数；
- CPU、操作系统、Python/MuJoCo/NumPy/OSQP 版本、线程数、亲和性和编译选项；
- 本次使用的 Python/C++/构建脚本**完整只读副本**及原生 DLL；
- 每个配置、源码、DLL、输入和结果文件的 SHA-256；
- `manifest.json` 中的文件数量、字节数、哈希和生成脚本哈希。

只保存源码哈希而不保存源码副本不再合格。结果目录创建后禁止覆盖；审计和图表
写入同级 `audits/`、`tables/`、`figures/`，不得回写原始日志。

## 3. 在线闭环实验必须记录的内容

### 3.1 相机与因果帧流

对每个请求帧和接受帧记录：

- `camera_id`、请求/曝光/接收时间戳、请求周期和曝光时 `q`；
- `T_world_camera`、内参、深度单位、有效像素掩码和近/远裁剪；
- 压缩原始深度帧，或足以无损重建该深度帧的数组；
- 从该帧生成的世界点、逐点 `U` 和逐点 `delta`；
- 丢弃、覆盖、合并、过期及因队列拥塞未处理的原因；
- 相机帧 → occupancy → CenterVox → proxy → controller 的版本链和延迟。

两相机数据进入一个世界地图。`camera_id` 只用于追溯，不得成为代理分组键。

### 3.2 free/occupied/unknown 地图与 CenterVox

每个地图发布版本必须保存：

- `publish_cycle/source_cycle/source_frame_ids`；
- free/occupied/unknown 体素的增量状态变化及周期性完整检查点；
- 每个 occupied CenterVox 的稳定 ID、世界坐标 `p_i`、`U_i`、`delta_i`；
- 生成它的原始点/相机帧 ID，融合次数和最后观测时间；
- 删除、转为 free、保持 unknown 或过期的原因；
- 按有序数组计算的完整 CenterVox SHA-256。

必须能只用保存数据从任意检查点重放出每一次控制使用的地图；不得依赖未来帧、
解析障碍物标签或未记录的内存状态。

### 3.3 在线代理发布

每个代理版本都保存：

- 共同 CenterVox 输入哈希和代理构造配置哈希；
- 代理 ID、中心、球半径，或椭球 `Q_core`、`U_proxy`、`delta_proxy`；
- 椭球轴长/方向、外包 AABB、MVT level/cell；
- 每个 CenterVox 的 owner、所有覆盖代理及覆盖余量；
- 每个已选代理的唯一见证 CenterVox；
- 候选数、贪心选择数、反向删除数、最终数、完全包含数和重复中心数；
- 显示快照 ID 与控制快照 ID，二者必须相同。

所有发布版本保存最终已选代理。下列关键版本还必须保存**全部候选和覆盖矩阵**：

- 初始发布；
- 两种表示首次产生控制差异前的版本；
- 首个 NEAR/CONTACT、首个 QP 失败或明显停滞版本；
- 首次严格成功版本；
- 最终版本。

### 3.4 每个 50 Hz 控制周期

`cycles.csv` 至少包含：

- cycle/time、活动 map/proxy 版本、源周期和快照年龄；
- `q/qdot`、末端位置、目标、误差、期望任务速度；
- proposed/executed `qdot`，二者不同的原因和回退类型；
- raw/broadphase/active pair 数及 limiting robot/proxy ID；
- QP 状态、迭代数、原始/对偶残差、QP 行哈希和 warm-start 命中；
- mapping、proxy、MVT、AABB、精确距离、QP 装配、求解和总周期耗时；
- deadline miss、接触/穿透、连续 sweep 结果；
- 严格 `<1 mm` 连续保持计数。

`pair_states` 对每个活动 pair 保存：

- robot/proxy ID、最近点或支持法向 `n`；
- 原始证书间隙 `c`、硬间隙 `h=c-d_safe`；
- NORMAL/NEAR/CONTACT；
- hard row、NEAR soft penalty、CONTACT recovery 是否加入；
- 椭球 Newton/二分次数、热启动来源和 KKT 残差。

失败时额外保存第一失败周期完整 QP 矩阵、上下界、目标向量、行语义 ID、求解器
状态和限制对几何，确保可独立做“只求可行性”复核。

### 3.5 防作弊与结果摘要

每次运行显式记录并验收：

- `path_planner=null`、无 RRT/A*/GCS/人工路点；
- `random_dither=false`，代理构造不读取目标和机器人受阻位置；
- 不使用真值碰撞回退、不缩短机器人证书、不减小安全余量；
- 不预载全局地图、不读取未来帧；
- Viewer 开关只改变显示，不能改变控制，也不能隐藏 UR5 本体。

摘要同时报告成功与失败：最终/最小误差、首次成功及保持周期、第一失败周期、最长
零速度段、状态比例、代理数、观测 late/never、接触/穿透和所有耗时 p50/p95/p99。

## 4. 同快照离线配对必须记录的内容

### 4.1 完整共同输入

每个配对输入保存为不可变 `input_centervox.npz`：

```text
voxel_ids          int64[N]
points             float64[N,3]      # p_i
uncertainty_shapes float64[N,3,3]    # U_i
offsets            float64[N]        # delta_i
states             int8[N]
source_frame_ids   ragged/int64
last_seen_cycles   int64[N]
```

还要保存源在线 run、map/proxy generation、源 `q`、相机位姿和输入 SHA-256。
球与椭球结果中的 `input_sha256` 必须逐字相同，否则配对立即失败。

为降低数据来源轨迹偏差，每个正式场景优先做交叉回放：

| 输入快照来源 | 构造球 | 构造椭球 |
|---|---:|---:|
| 球型在线运行 | 是 | 是 |
| 椭球型在线运行 | 是 | 是 |

这只是两次在线运行加四次离线构造，不是四次闭环。

### 4.2 构造器与代理结果

球/椭球构造分别保存在独立只读目录，但必须共享同一个输入文件。每组保存：

- 完整构造器源码、配置、依赖和原生二进制；
- 全部候选参数 `all_candidates.npz`；
- CenterVox—候选覆盖关系的稀疏 CSR 矩阵；
- 每条覆盖边的几何余量；
- 在线贪心选择顺序、反向删除顺序和最终 selected IDs；
- 每个最终代理的唯一见证；
- 覆盖失败、重复中心、包含代理及半径/轴长/法向突出量分布；
- 球解析距离或椭球真实最近点/支持平面残差。

球和椭球允许数量不同，但必须使用相同输入证书、尺度上限、安全定义和候选生成中
不依赖目标的规则。不得为了得到想要的结论而单独调整一种表示的体素或误差界。

### 4.3 固定候选族的最少覆盖

对完整候选覆盖矩阵离线求解：

\[
\min_{x_j\in\{0,1\}}\sum_jx_j,
\qquad
\sum_{j:i\in C_j}x_j\ge1,\;\forall i.
\]

保存模型文件、求解器名称/版本、线程、seed、时间限制、最好可行解、最好下界、
MIP gap、终止原因和 selected IDs。只有 gap=0 才称为“固定候选族全局最少”；
否则报告最优性区间。不得把它写成连续球心/半径空间的全局最少。

在线控制仍使用有时限的贪心+反向删除，不等待整数规划。离线最优只用于量化在线
结果距固定候选族最少值有多远。

### 4.4 截面和性能配对

每个共同输入同时保存：

- 必经截面的几何定义及所选机器人球为什么必须穿越；
- 全截面连续 1-Lipschitz/区间分支定界，而不只采样停止点；
- 球的连续最大安全余量上界、椭球的连续开口下界或 open witness；
- 网格仅用于画图，证明必须包含误差界和终止公差；
- full scan、MVT-scalar、MVT-AVX2、精确窄阶段和 QP 行构造的原始计时样本；
- 所有候选 ID、顺序、精确行和命令的 oracle 等价性；
- 硬件、亲和性、预热次数、重复次数和当前源码/DLL 哈希。

离线性能数字必须标明“offline paired snapshot”；端到端实时仍由在线闭环日志证明。

## 5. 推荐目录结构

```text
formal_results/<scene_id>/
├── online/
│   ├── sphere/<run_id>/
│   └── ellipsoid/<run_id>/
├── paired/
│   ├── source-sphere/<snapshot_id>/
│   │   ├── input_centervox.npz
│   │   ├── sphere/
│   │   ├── ellipsoid/
│   │   └── pair_result.json
│   └── source-ellipsoid/<snapshot_id>/...
└── frozen_manifest.json
```

每个 `sphere/ellipsoid` 配对目录至少包含：

```text
all_candidates.npz
coverage_matrix.npz
selected_proxies.npz
coverage_audit.json
minimum_cover_result.json
cross_section_certificate.json
query_benchmark.json
source_snapshot/
manifest.json
```

## 6. 进入下一阶段前的记录门

1. **R0 配置锁定**：场景、相机、控制、代理和成功门已写入配置并哈希。
2. **R1 源码冻结**：完整源码/DLL/环境已复制，不能只保存哈希。
3. **R2 在线因果重放**：任意控制周期可还原其 map/CenterVox/proxy 版本。
4. **R3 关键候选可重建**：关键发布版本保存全部候选和覆盖矩阵。
5. **R4 同输入配对**：球/椭球 `input_sha256` 完全一致，覆盖均通过。
6. **R5 最少覆盖边界**：报告固定候选族最优值或带下界的 MIP gap。
7. **R6 几何/查询等价**：连续截面和 full-scan oracle 审计通过或如实失败。
8. **R7 制品冻结**：结果目录 manifest 全文件哈希复核通过。

任一门失败就记录 limitation；不得先修改场景、延长时间或重跑到出现期望结果。

## 7. 代码审核和讲解顺序

后续代码审核按下列数据链进行，每个模块都讲清“输入—公式—输出—不变量—失败
行为—保存制品”，不从 Viewer 或总运行脚本倒推实现。

| 顺序 | 模块 | 主要文件 | 审核重点 |
|---:|---|---|---|
| 1 | 场景与机器人证书 | `protocol_drawer_scene.py`, `model.py`, frozen `scene.xml` | 目标、q0、真实几何、65 球证书，不存在长探杆/夹爪作弊 |
| 2 | 两台移动深度相机 | `depth_camera_perception.py` | 相机实体位姿、第一可见表面、世界变换、逐点 `U_i/delta_i` |
| 3 | 因果占据地图 | `native_occupancy.py` | free/occupied/unknown、时间版本、无未来帧、两相机融合为一图 |
| 4 | CenterVox 与候选生成 | `incremental_proxy_manager.py` | 7.5 mm 规则、source ID、覆盖、候选族、禁止按相机重复 |
| 5 | 球/椭球拟合 | `pointcloud_proxy.py` | 球半径公式、`Q_core+U+delta`、短轴不重膨胀、不可约选择 |
| 6 | 多级体素表与 SIMD | `native_mvt.py`, `native_mvt/native_mvt.cpp` | size-selected level、每层 27 格、SoA/AVX2、full-scan 无漏检 |
| 7 | 椭球精确法向 | `native_ellipsoid_support.py` 及原生核 | safeguarded Newton、必要时二分、热启动、KKT 残差 |
| 8 | LiuQP 三状态 | `protocol_liuqp_controller.py` | `c/h` 区分、NORMAL/NEAR/CONTACT、hard/soft/recovery、零回退 |
| 9 | 异步调度和日志 | `run_protocol_v3_async_online.py` | frame→map→proxy→control 因果链、线程、快照原子发布、计时口径 |
| 10 | Viewer 与独立审计 | viewer/verify/audit scripts | 显示等于控制快照，开关不改控制；覆盖、截面、MVT、哈希独立复核 |

审核每个模块时同时打开对应冻结输入和一条最小可复算测试；仅阅读源码而没有输入
输出对照，不视为通过。
