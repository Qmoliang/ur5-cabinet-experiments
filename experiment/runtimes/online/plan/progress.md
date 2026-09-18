# 进度

## 2026-09-02 椭球型 LiuQP 数学推导

- 阶段：S2 方法定义 + S4 分章节写作。
- 用户指定 `output/documents/LiuQP_Mathematical_Derivation_and_Theory_Guide_English_Expanded.docx` 为母版，要求复用机械臂运动学、模块点速度和末端任务推导，从障碍物椭球开始重建后续公式。
- 已生成 `chapters/02_椭球型LiuQP数学推导.md`。公式（1）—（20）沿用母版；椭球最近点 KKT、标量 Newton/必要二分/热启动、一般支持和的 Riemannian Newton、切平面、速度硬行、ordered erase-remove 和三状态 QP 已完整推导。
- 双重审查记录保存于 `plan/review/chapter-02-ellipsoidal-liuqp-derivation-review.md`。文档结构为 13 节、73 个唯一公式标签、16,529 个字符，SHA-256 为 `6FA99712196D6AB6803596796DE3A217BB95CCEEE604381B026AC25C0358706C`。
- 独立数值特例通过球极限、表面方程、法向对齐、支持点恒等式、标量根残差和间隙导数检查；9 项 Python 椭球回归测试与 13 项原生支持核测试全部通过。
- 当前等待用户逐段确认第二章，尤其是“公式（1）—（20）复用、公式（21）后开始椭球重推”的继承边界。

### Capability-use audit

- Required skills: using-research-writing、paper-orchestration、writing-chapters、writing-core、documents、verification。
- Skills actually used: 上述六项；documents 用于只读提取源 DOCX 结构与公式，writing-chapters 用于单章交付，verification 用于最终结构、数值与测试审计。
- Inputs consumed: 用户指定的 English Expanded DOCX、其精确生成源码、`ellipsoid_model.py`、`protocol_liuqp_controller.py`、`native_mvt/native_mvt.cpp` 与现有椭球测试。
- Inputs not used and why: 正式 v4.3/v4.4/v5 数值结果未写入本章，因为本章只建立算法公式；在线文献检索未使用，因为当前任务是按用户给定母版和已冻结实现整理，不是相关工作写作。
- Artifacts produced: 第二章 Markdown、章节任务单、规格/质量审查记录、进度与研究记录更新。
- Verification run: 结构/公式标签/占位/符号扫描、研究写作风格检查、独立几何数值特例、9+13 项实现测试、SHA-256。
- Remaining risk: 源 DOCX 因本机缺少 LibreOffice 未完成逐页视觉渲染；内部穿透点分支仍需在代码审核章明确为恢复保护；本章尚待用户确认，不能视为全文冻结。

## 2026-09-01 原理与代码审核文档启动

- 阶段：S2 方法定义 + S4 分章节写作。
- 用户要求从第一性原理开始，以中文 Markdown 逐段整理到 Word；沿用此前已确认的技术规格题目、研究边界和十部分结构，不重新询问已冻结信息。
- 当前中等规模任务仅覆盖 `chapters/01_第一性原理与问题定义.md`，不把实验版本结果写入原理章。
- 已建立 `plan/task-packets/chapter-01-first-principles.md`；章节论证链固定为真实安全集合、连续代理证书、球/椭球支持半径、真实最近法向、速度级硬约束、LiuQP 三状态及 VCC-inspired 等价查询。
- 第一章规格审查和质量审查已记录于 `plan/review/chapter-01-first-principles-review.md`；当前状态为可交付用户逐段审阅，尚未进入第二章。

### Capability-use audit

- Required skills: paper-orchestration、brainstorming-research、writing-chapters、writing-core、verification。
- Skills actually used: 上述五项；brainstorming-research 沿用 `project-overview.md` 与 `outline.md` 中已经记录的用户确认，不重复提问。
- Inputs consumed: 项目概览、提纲、进度、研究记录、数学规格、实验记录契约、当前 LiuQP 控制器的三状态与 QP 行实现。
- Inputs not used and why: v4.3/v4.4 数值结果未使用，因为本章只定义原理；文献在线检索未使用，因为本章尚未撰写引言或相关工作，原始来源逐式核验留给独立章节。
- Artifacts produced: 第一章 Markdown、章节任务单、规格与质量审查记录。
- Verification run: 文件/字符数/标题/公式编号检查；占位与风格词检查；实现常量和法向符号核对；各向同性椭球退化为球的数值特例；空白与冲突标记检查。
- Remaining risk: 式（14）的含不确定性支持和需要在专门数学推导章给出单位球面优化与求解器细节；本章仍等待用户逐段确认，不能视为全文冻结。

## 2026-08-28

- 阶段：S3 实验协议与基线审计。
- 已确认：Eq. (17) 不提供整条路径；`model.py` 的三段 Cartesian 路点由本地复现代码人工设定。
- 已运行未修改的 `shelf --guidance direct`：纯最终目标 LiuQP 成功，最终误差 16.87 mm。
- 已有 `cage --guidance direct`：纯最终目标 LiuQP 成功，最终误差 20.49 mm。
- 因此当前两个 UR5e 场景不能作为“真实 QP 卡住”的证据；需新增或筛选可复核失败查询。
- 当前任务：实现并验证球/椭球证书对照，旧结果保持原路径不覆盖。
- 已完成固定 204 障碍代理主对照：球版失败（203.03 mm），完整机器人/障碍物椭球版成功（14.97 mm），两组 QP 可解率 100%。
- 已完成固定层级闭合证书：关键末端球的球模型开口 -35.62 mm，完整椭球模型约 +254.0 mm；该证书不外推到其他球树分辨率。

## 2026-08-29 增量感知抽屉规格

- 阶段：S2 方法定义 + S3 实验协议，尚未启动新场景实验。
- 用户已确认技术规格题目与边界。
- 已否决长探杆、控制开始前完整融合点云、渲染时隐藏机器人导致的穿透式感知，以及任何人工路径或区域链引导。
- 当前任务：冻结完整数学定义、模块输入输出、实验因果对照与实时性验收标准；完成并验证 Word/Markdown 后才允许实现。
- 已完成 12 组固定层级联合小扰动审计：0/12 球版成功，所有 QP 可解率 100%。
- 已完成球树分辨率审计：255/510 球仍失败，969 球成功（17.56 mm）；主结论修正为代理效率而非绝对不可达。
- 已保留 x=0.70 远目标及早期候选负结果；结论不外推到一般同伦/姿态局部极小。
- 已生成最终主图、技术说明与 13 项通过的单元测试。
## 2026-08-29 文档冻结与实验实现启动

- 完成 747 行 Markdown 数学规格与 18 页 Word 版；Word/PDF 结构核验为 59 个标题、46 个公式段、3 张表，逐页栅格边界无裁切。
- 新增 `incremental_drawer_frozen_v1`：120 mm 紧凑夹爪、单个 D405 量级腕部相机外壳，初始末端在抽屉外且低于开口，初始接触数为 0。
- 新增 `occluding_self_filter=True` 感知模式：机器人参与深度遮挡，渲染后按 geom ID 删除自身返回；旧实验默认行为保持不变。
- 原有 18 项回归测试通过；新增 3 项协议测试通过。
- 运行了全表面 oracle 开发预检（明确不得作为论文实验）：20 s 时球/椭球均未到达；球停在前沿外侧约 x=0.326 m，椭球到 x=0.356 m，说明场景 v1 的主对照尚未形成合格结果，不能宣称成功。
- 下一步：实现 free/occupied/unknown 增量地图与在线代理局部更新；在不改已冻结协议的前提下完成可达审计和结构屏障审计，再决定是否以版本化场景修订替换 v1。

## 2026-08-30 正式 v3/v5 双移动相机实验

- 冻结 `formal_drawer_two_camera`：UR5 无夹爪/长杆，65 个机器人证书球，腕部与前臂两个带实体外壳的局部深度视角，唯一 Cartesian 目标，50 Hz 控制。
- 静态门槛通过：2 个无碰撞目标 IK；无障碍同一 LiuQP 2.96 s 到达；球证书连续截面上界 -5.827 mm；椭球同截面全 pair 余量 +5.034 mm；两类代理都覆盖 200 个匹配输入单元。
- v5 最终在线因果组：球型运行 45 s、2250 周期失败，最终误差 355.705 mm；椭球型 12.26 s 首次到达，623 周期后最终误差 17.148 mm。两组 QP 命令均未被规划器、dither、真值回退或 sweep guard 修改，MuJoCo 精确穿透周期均为 0。
- 因果审计通过：球 25 个、椭球 13 个三态地图快照与在线 free/occupied 计数逐帧一致；候选暴力 AABB 重建与在线计数逐周期一致；连续代理 sweep 重放均为 0 个 unsafe 周期。
- 查询加速等价性通过：球 146250 次、椭球 24830 次查询中，AABB-Brute、5 层 MVT-Scalar 和 MVT-AVX2 的候选 ID/顺序、QP 行、pair 状态与轨迹完全一致。
- 实时门槛通过：控制计算 p99 为球 8.716 ms、椭球 16.355 ms。球和椭球各有 2 个计算周期超过 20 ms，因此只声明冻结的 p99 软实时，不声明硬实时。
- 修复 C++ 零热启动未归一化缺陷；新增 \(10^{-7}\) KKT 重启/硬门。v5 椭球最大 KKT 残差 \(3.36\times10^{-8}\)，旧 v4 结果未删除。
- 原生 SIMD 微基准：约 500 个正式代理时 AVX2 中位数快 1.38%；15480 个未聚合 CenterVox 代表点压力输入时快 6.64%。严格等价完整控制 p99 中 SIMD 未快于标量，不能宣称显著整体 SIMD 加速。
- CenterVox 5/7.5/10 mm 敏感性汇总完成，只有 7.5 mm 椭球组通过全部门。0.3 mm 有界噪声探索组因证书/代理膨胀失败且 sweep、地图重放门未全过，保留为负面结果而不进入主证据。
- 综合证据清单为 `formal_results/final_two_camera/formal_protocol_v3_evidence_manifest_v5.json`。
- 已从 v5 实际日志生成两张 450 DPI PNG/SVG 正式图、三张数据表和两段因果 MuJoCo 对照视频；各自清单保存源文件、脚本和输出 SHA-256，表格声明 `manual_values=false`。视频只回放已保存控制历史与因果快照，不重跑控制器。
- 交互式 MuJoCo viewer 已切换至 v5；可切换球/椭球、点云、全部代理、候选代理、QP 激活代理、地图与机器人证书球，且隐藏叠加层不会隐藏 UR5 或抽屉本体。
- 主冻结场景的正式证据、论文主图表和对照视频已齐备；整个研究仍未达到“噪声鲁棒性完整交付”，因为 0.3 mm 有界噪声组未到达且连续 sweep、地图重放未全部通过。下一算法版本必须先讨论并冻结保守代理在线合并/不确定性处理，再做多随机种子实验，不能把负结果改名为成功。


## 2026-09-15 环境已知体积实验代码阅读指南

- 阶段：S2 方法定义 + S4 分章节写作。
- 已生成 `chapters/03_环境已知体积实验代码阅读指南.md`，把第二章数学对象重新组织为 `run.py -> controller.solve -> native_support -> collision.cpp -> OSQP -> Euler 更新` 的单周期因果链。
- 章节明确区分当前默认 C++ 融合热路径、Python 数学参考/回退路径和 \(U_j\) 方向不确定性未启用分支；单独解释 `ellipsoid_closest_prune_pairs_warm` 中 CONTACT 只用于恢复 pair 优先保留，最近点仍由式（E-B6）的 KKT 标量根计算。
- 双重审查记录保存于 `plan/review/chapter-03-known-volume-code-reading-guide-review.md`。章节为 26,927 个字符、736 行、13 个二级标题、30 个三级标题，SHA-256 为 `08850FAA96E893582A6257F0BF9880E0B02C5D3DCD83549DEDCDEC74E22A116C`。
- 当前状态：可交付用户阅读和修改，尚未按用户反馈冻结。

### Capability-use audit

- Required skills：using-research-writing、paper-orchestration、writing-chapters、writing-core、verification。
- Skills actually used：上述五项。
- Inputs consumed：现有项目概览、大纲、进度和研究记录；第二章 73 个公式标签；known-volume 的场景、完整体积构造、机器人证书、MVT、Python 几何参考、ctypes 绑定、控制器、C++ native 内核、运行/聚合/回放/交付测试及真实结果汇总。
- Inputs not used and why：增量深度点云 v5 的相机、CenterVox、在线地图和正式场景脚本未进入正文，因为本章范围是用户当前阅读的独立 known-volume 实验；外部文献未新增，因为本章做本地数学—实现追踪，不写相关工作。
- Artifacts produced：第三章 Markdown、章节任务单、双阶段 review、大纲和进度/研究记录更新。
- Verification run：文件与结构统计、公式/关键内核计数、Markdown 围栏检查、占位和禁用表达扫描、当前源码关键调用点定位、`tests/check_experiment.py`。
- Remaining risk：未重新编译 DLL 证明当前 CPP 与二进制逐位对应；`scene.json` 证书盒与 `drawer.xml` 物理 geom 的一致性仍由冻结资产契约承担；章节尚待用户确认。





## 2026-09-16 第四、第五章实验版本证据审计

- 阶段：S3 实验证据审计，尚未开始第四、第五章正文。
- 已核对 v4.3/v4.4 冻结结果、v5.0 总协议、v5.1/v5.2/v5.3 冻结协议和逐运行 `v5_run_complete.json`/`summary.json`。
- v5.1 确有球型五次和椭球型五次正式对照。椭球任务成功 5/5、球任务成功 0/5；但椭球只有 1/5 通过所有非任务证据门，主要缺口为四次 `late=1` 和 r05 两次 deadline miss。因此不得把 5/5 任务到达写成 5/5 完整协议通过。
- v5.2 完成球/椭球各五次重跑：球 0/5、椭球 3/5；可观测性修复但任务重复性和 deadline 门仍未全部通过。
- v5.3 保存椭球三次、球四次后按协议停止；方向子证书使椭球代理约增至 3,300，并出现固定时长边界与实时尖峰，属于诊断负结果。
- 已建立 `plan/task-packets/chapter-04-05-experiment-guide.md` 和 `plan/review/chapter-04-05-version-evidence-audit.md`；正文建议采用 4.3 v4.3、4.4 v4.4、5.1 v5.1 成对重复、5.2 v5.2、5.3 v5.3 的结构。

