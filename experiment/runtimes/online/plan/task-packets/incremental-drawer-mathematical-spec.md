## Task Packet

- Scope: 编写增量深度感知 UR5 抽屉进入任务的完整数学推导与无作弊实验协议。
- Files to read: LiuQP 论文与控制器代码、VCC 论文、FastIRIS 论文、现有点云/椭球/MVT 实现、用户确认的任务边界。
- Files allowed to edit: 本任务 plan 文件、技术规格 Markdown、技术规格 DOCX 及其生成脚本；不得运行或修改新实验场景。
- Required skills: paper-orchestration, writing-chapters, writing-core, experiment-results-planning, documents, verification。
- Evidence/data inputs: 三篇本地论文、MuJoCo/UR5e 模型定义、现有真实失败结果只用于识别反模式，不写作新实验结果。
- Required artifacts: 完整数学规格 Markdown、渲染验证后的 DOCX、实验协议、方法—实验可追溯表、表/图数据契约、capability-use audit。
- Rejection checks: 不出现长探杆；不使用预融合全图；不隐藏机器人后让深度射线穿透；不使用人工路点、A*/RRT/GCS、IRIS 区域链或随机解锁；不把任务空间通路误当全机械臂构型可达；不把 Python 字典 MVT 称为真实 VCC；不把 QP 求解时间冒充完整控制周期。
- Validation commands: Markdown 结构/公式符号检查；禁用词与禁止方案检索；DOCX 渲染逐页检查；标题、表格、公式越界检查；输出文件存在与页数检查。
