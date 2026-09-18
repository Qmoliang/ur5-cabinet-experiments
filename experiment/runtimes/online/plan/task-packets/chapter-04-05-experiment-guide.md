# Task Packet：第四、第五章增量感知实验解读

- Scope：在第三章已知体积代码阅读指南之后，自上而下解释增量深度感知实验；重点覆盖第四章 4.3、4.4 与第五章 5.1，并把实验问题、算法改动、控制变量、结果、失败门和允许结论连成一条证据链。
- Stage：S3 实验证据审计 + S4 分章节写作。
- Files to read：第二章数学推导、第三章代码阅读指南、`V4_3_NO_CORE_REINFLATION_FORMAL_RESULTS.md`、`V4_4_ADAPTIVE_IRREDUNDANT_SPHERE_RESULTS.md`、v4.3/v4.4 冻结有限证据、v5.0 总协议、v5.1/v5.2/v5.3 冻结协议、各次 `v5_run_complete.json` 与 `summary.json`、正式表格、验证脚本和复现入口。
- Files allowed to edit：第四、第五章 Markdown、本任务单、对应 review、`plan/outline.md`、`plan/progress.md`。
- Required skills：using-research-writing、paper-orchestration、writing-chapters、writing-core、verification。
- Required argument chain：已知体积实验的不足 → 因果深度感知与 CenterVox → v4.3 取消核心椭球二次增厚 → v4.4 构造可信球基线 → v5.0 重定义公平性 → v5.1 球/椭球五次成对重复 → v5.2 延迟修复负结果 → v5.3 方向子证书诊断与停止原因。
- Rejection checks：不得把 v4.3/v4.4 说成同一最终 CenterVox；不得把球停滞说成整个必经截面被证明封死；不得把 v5.1 写成椭球单组实验；不得把 5/5 任务到达写成 5/5 完整协议通过；不得把 v5.3 写成已经完成的最终版本；不得用早期 `table1_main_online` 的 623/2250 周期演示结果替代 v5.1 的 3000 周期正式重复。
- Validation：逐运行 JSON 重新聚合；核对协议预注册矩阵、结果目录数量、共同参数、成功门、感知门、实时门、穿透与 sweep；章节完成后进行公式—代码—结果三向追踪和 Markdown/风格检查。

