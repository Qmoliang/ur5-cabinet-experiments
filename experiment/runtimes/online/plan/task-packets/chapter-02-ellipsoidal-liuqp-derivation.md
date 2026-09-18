## Task Packet

- Scope: 以用户指定的 `LiuQP_Mathematical_Derivation_and_Theory_Guide_English_Expanded.docx` 为结构和符号母版，撰写中文“椭球型 LiuQP 数学推导”。复用原稿中机械臂图运动学、末端/模块点速度、反馈控制、关节界和软任务推导；从障碍物表示开始逐式推广为椭球。
- Files to read: `output/documents/LiuQP_Mathematical_Derivation_and_Theory_Guide_English_Expanded.docx`、`tools/build_liuqp_theory_guide.py`、`chapters/01_第一性原理与问题定义.md`、`ellipsoid_model.py`、`protocol_liuqp_controller.py`、`native_mvt/native_mvt.cpp`、`plan/experiment-recording-and-code-audit-contract.md`。
- Files allowed to edit: 本任务单、`chapters/02_椭球型LiuQP数学推导.md`、本章审查记录、`plan/progress.md` 和 `plan/notes.md`。
- Required skills: using-research-writing、paper-orchestration、writing-chapters、writing-core、documents（只读母版）、verification。
- Evidence/data inputs: 母版生成脚本中的公式（1）—（27）与补充式 A—I；Word 只读统计 691 段、407 个 OMath；当前标量最近点 Newton/二分实现；当前原生受保护 Riemannian Newton 支持和实现；当前 LiuQP 三状态和 QP 行。
- Required artifacts: 一份可逐段转入 Word 的中文 Markdown；原球形公式与椭球对应式之间有明确映射；完整给出最近点 KKT、支持函数、分离平面、间隙导数、冗余代理删除、综合 QP 和三状态。
- Rejection checks: 不重新发明或改变母版公式（1）—（20）的运动学逻辑；不使用中心连线代替椭球真实最近法向；不引入 `ellipsoid-fast`；不把标量 Newton/二分与一般支持和 Riemannian Newton 写成同一 pair 的重复步骤；不把安全余量重复计入代理；不把 IRIS-inspired 分离平面写成完整 IRIS/MVIE/GCS；不声称椭球消除所有局部最优。
- Validation commands: 标题、公式编号、定界符、占位符和禁用词检查；球退化特例 `Q=R^2I`；最近点 KKT 残差数值检查；支持函数与显式支持点一致性检查；有限差分验证间隙导数；源码默认阈值与法向方向核对。

## Chapter writing contract

- Target chapter file and exclusive owner: `chapters/02_椭球型LiuQP数学推导.md`，当前主任务独占编辑。
- Required argument chain: 母版不变部分 → 障碍椭球证书 → 球—椭球最近点 KKT → 精确分离平面 → 支持和一般式 → 间隙导数与线性速度行 → 椭球支配删除 → 综合 QP → NORMAL/NEAR/CONTACT → 周期级重建与算法边界。
- Minimum prose length: 8000 个中文字符，不含本任务单。
- Required sources and data artifacts: 指定 DOCX、其生成脚本、上述三个当前实现文件；不使用旧 v1—v12 的实验结果支撑数学结论。
- Prohibited structure: 不把正文写成代码审计流水账；不使用实验胜负替代推导；不保留 TODO 或待补公式；不大量复制英文原文。
- Required handoff: 报告章节路径、复用范围、重推范围、验证证据、未进入本章的 VCC/CenterVox 内容和任何实现—理论差异。
