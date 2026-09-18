# Task Packet：第三章 known-volume 代码阅读指南

- Scope：为 `D:\MuJoCo\LiuQP_known_volume` 编写一章自上而下的代码阅读指南，把第二章数学推导连接成一个可执行实验流程。
- Stage：S2 方法定义 + S4 分章节写作。
- Files to read：第二章数学推导；`run.py`；`assets/scene.json`；`src/known_volume.py`；`src/core/{robot,voxel_index,geometry,native_support,controller}.py`；`native/collision.cpp`；`aggregate.py`；`view.py`；`tests/check_experiment.py`；真实结果汇总。
- Files allowed to edit：`chapters/03_环境已知体积实验代码阅读指南.md`、本任务单、`plan/outline.md`、`plan/progress.md`、对应 review 文件。
- Required skills：using-research-writing、paper-orchestration、writing-chapters、writing-core、verification。
- Required argument chain：实验输入与公平对照 → 程序入口 → 静态初始化 → 单控制周期 → 候选查询 → 精确几何 → ordered erase-remove → 三状态 QP → Euler 更新与 MuJoCo 审计 → 日志、聚合与验证。
- Required artifacts：模块依赖图、周期时序图、文件职责表、第二章公式到代码的双向索引、当前默认分支说明、`ellipsoid_closest_prune_pairs_warm` 展开说明、推荐阅读顺序和断点观察表。
- Rejection checks：不得把 MVT 当作距离求解器；不得把 CONTACT 参数当作最近点求解方法；不得把 Python 参考路径说成当前默认热路径；不得把 MuJoCo 接触审计说成 QP 命令来源；不得把单最终目标说成路点路径；不得把 dormant uncertainty 分支说成当前实验已启用。
- Validation commands：文件/标题/公式引用统计；关键模块与函数名存在性扫描；代码行号抽查；禁用词和占位符扫描；`tests/check_experiment.py` 回归检查。

