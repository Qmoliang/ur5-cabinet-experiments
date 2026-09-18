## Task Packet

- Scope: 只撰写中文方法文档的第一章“第一性原理与问题定义”，建立真实碰撞、点云证书、球/椭球间隙、速度级 QP 和查询加速之间的因果链；本章不报告实验胜负。
- Files to read: `plan/project-overview.md`、`plan/outline.md`、`plan/notes.md`、`UR5_incremental_depth_LiuQP_ellipsoid_VCC_math_spec.md`、`plan/experiment-recording-and-code-audit-contract.md`、`protocol_liuqp_controller.py`。
- Files allowed to edit: 本任务单、`chapters/01_第一性原理与问题定义.md`、`plan/progress.md`。
- Required skills: paper-orchestration、brainstorming-research（沿用已确认范围）、writing-chapters、writing-core。
- Evidence/data inputs: 已冻结数学规格；当前控制器中 `c/h`、NORMAL/NEAR/CONTACT、硬碰撞行、NEAR 软惩罚和 CONTACT 恢复行的实际定义；2026-09-01 记录协议中在线闭环与同快照离线配对的区分。
- Required artifacts: 一份可直接逐段转入 Word 的中文 Markdown；公式的法向约定、原始间隙和安全间隙必须自洽；明确 VCC-inspired 结构不改变 QP 几何。
- Rejection checks: 不把 IRIS 写成在线路径规划器；不写 RRT/A*/GCS/人工路径；不把任意支持方向当作椭球真实最近法向；不把 NEAR 软惩罚写成软化碰撞约束；不声称椭球消除所有局部极小；不把 v4.3/v4.4 的局部停滞扩写为全局截面闭塞证明。
- Validation commands: 检索禁用规划器和过度结论；核对公式中法向方向；核对代码默认阈值 `d_safe=0.006 m`、`d_near=0.040 m`、`d_contact=0`；检查 Markdown 标题与公式定界符。

## Chapter writing contract

- Target chapter file and exclusive owner: `chapters/01_第一性原理与问题定义.md`，当前主任务独占编辑。
- Required argument chain: 任务不是路径跟踪 → 真实安全集合不可直接在线求解 → 点云需要连续外包证书 → 球的各向同性保守性可能制造假闭塞 → 椭球的方向支持半径保留狭窄通道 → 真实最近法向产生速度级硬约束 → LiuQP 三状态不改变硬安全底线 → VCC-inspired 结构只做等价加速。
- Minimum prose length: 3000 个中文字符；不含本任务单。
- Prohibited structure: 不用实验版本流水账代替原理；不把正文写成条目堆叠；不保留“待补”“TODO”等占位符。
- Required handoff: 报告文件路径、公式核对结果、尚未进入的下一章主题和任何仍需用户判断的表述。
